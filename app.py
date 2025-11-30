import os
import io
import time
import tempfile
import traceback
import threading
import re
from dotenv import load_dotenv

# Import for Telegram and Flask
import telebot
from flask import Flask, request

# Import for FFMPEG (Video processing) and Groq (Transcription/LLM)
import ffmpeg
from groq import Groq
from groq.types.chat import ChatCompletion

# Load environment variables (used for local testing, Render uses its own Environment variables)
load_dotenv()

# --- Configuration ---
BOT_TOKEN = os.environ.get('TELEGRAM_TOKEN')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
ADMIN_USER_ID = os.environ.get('ADMIN_USER_ID') # Optional: For admin alerts
FFMPEG_TIMEOUT = 300 # 5 minutes timeout for FFMPEG 
LLM_MODEL = "llama3-8b-8192" # A smaller, more stable Groq LLM for translation

# Initialize Clients
try:
    # Use parse_mode='HTML' for better formatting
    bot = telebot.TeleBot(BOT_TOKEN, parse_mode='HTML')
    groq_client = Groq(api_key=GROQ_API_KEY)
except ValueError as e:
    print(f"FATAL ERROR: Failed to initialize TeleBot. Error: {e}")
    exit(1)
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize clients. Check environment variables. Error: {e}")
    exit(1)

# Initialize Flask App
app = Flask(__name__)

# --- Helper Functions ---

def safe_send_error_message(chat_id, error_message, full_traceback=""):
    """
    Sends an error message to the user, ensuring the text does not exceed 4096 characters.
    """
    full_message = f"❌ <b>שגיאה קריטית:</b> {error_message}\n\n"
    
    if full_traceback:
        MAX_TRACEBACK_LEN = 3000
        
        if len(full_traceback) > MAX_TRACEBACK_LEN:
            full_traceback = full_traceback[:MAX_TRACEBACK_LEN] + "\n... [המשך השגיאה קוצץ] ..."
        
        full_message += f"<u>פרטים טכניים:</u>\n<pre>{full_traceback}</pre>"
    
    try:
        # Use HTML parse mode for better formatting and code block support
        bot.send_message(chat_id, full_message, parse_mode='HTML')
    except telebot.apihelper.ApiTelegramException as e:
        bot.send_message(chat_id, f"❌ שגיאה קריטית: לא ניתן לשלוח את פרטי השגיאה. (שגיאה: {e})")

# FFMPEG Command: Subtitle burning and re-encoding
def burn_subtitles_fast(input_path, subtitle_path, output_path):
    """
    Uses FFMPEG to burn subtitles into the video file using the 'subtitles' filter.
    Alignment=2 (Bottom Center) for subtitles.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                # Alignment=2 sets subtitles to the bottom center (default is 10)
                # MarginV=20 sets padding from the bottom edge
                vf=f"subtitles='{subtitle_path}':force_style='Fontname=Noto Sans Hebrew,FontSize=28,Alignment=2,Outline=2,Shadow=1,MarginV=20'",
                vcodec='libx264',
                acodec='copy',
                pix_fmt='yuv420p',
                preset='ultrafast',  # Fast encoding speed
                crf=23, # Default quality for H.264
                strict='experimental'
            )
            .global_args('-t', '300') # Hard limit of 5 minutes (300 seconds)
            .run(overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFMPEG Encoding Failed. Stderr: {e.stderr.decode('utf8', errors='ignore')}")
    except Exception as e:
        raise e

def parse_srt_content(srt_content):
    """
    Parses the SRT string into a list of subtitle blocks (index, time, text).
    Returns: list of dicts: [{'index': 1, 'time': '00:00:00,000 --> 00:00:02,123', 'text': '...'}]
    """
    blocks = []
    # Regex to capture Index, Time (full line), and Text
    # This regex is designed to be robust against empty lines between blocks
    # Note: \s*? makes the match non-greedy
    srt_pattern = re.compile(r'(\d+)\s*?(\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3})\s*?(.*?)(?=\n\d+|\Z)', re.DOTALL)
    
    matches = srt_pattern.findall(srt_content)

    for index, time_str, text_block in matches:
        # Clean up the text block (remove extra newlines/spaces)
        clean_text = text_block.strip()
        if clean_text:
            blocks.append({
                'index': int(index),
                'time': time_str.strip(),
                'text': clean_text
            })
            
    return blocks

def generate_new_srt(translated_blocks):
    """
    Generates a new SRT string from the translated blocks.
    """
    new_srt = ""
    for block in translated_blocks:
        new_srt += f"{block['index']}\n"
        new_srt += f"{block['time']}\n"
        new_srt += f"{block['text']}\n\n"
    return new_srt.strip()


def get_transcript_and_translation(audio_data):
    """
    1. Transcribes audio to English SRT (to get time synchronization).
    2. Uses Groq LLM to translate the full transcript to Hebrew.
    3. Rebuilds the SRT file with the Hebrew translation.
    Returns: original_text (str), translated_srt_content (str)
    """
    # Create a temporary file to hold the audio data
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_name = temp_audio_file.name

    try:
        # --- 1. Transcription to English SRT (for sync) ---
        with open(temp_audio_file_name, "rb") as audio_file:
            # We request SRT here. It will be the original language (English) with correct timing.
            transcript_response_srt = groq_client.audio.transcriptions.create(
                file=(temp_audio_file_name, audio_file.read()),
                model="whisper-large-v3",
                response_format="srt", 
                language="en"
            )
            original_srt_content = transcript_response_srt
        
        # Parse the English SRT content to get the text blocks and timings
        original_blocks = parse_srt_content(original_srt_content)
        if not original_blocks:
             return None, None
        
        # Combine the original text from all blocks for the LLM translation
        original_text = " ".join([b['text'] for b in original_blocks])
        
        # --- 2. Translation using Groq LLM ---
        
        # System instruction to guide the LLM's output
        system_prompt = "You are a professional subtitle translator. Translate the following English transcript into high-quality, clear, and colloquial Hebrew. The output must ONLY contain the translated text, nothing else. Do not add any greetings, explanations, or context."

        # User prompt is the full original transcript
        user_query = f"Translate the following English transcript into Hebrew:\n\n---\n{original_text}\n---"
        
        translation_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model=LLM_MODEL,
            temperature=0.1
        )
        translated_full_text = translation_response.choices[0].message.content.strip()

        # --- 3. Re-Segmentation (The Smart Part) ---
        # The LLM gives us one block of translated text. We need to split this new Hebrew
        # text back into the original timings (blocks) to maintain sync.

        # Simple approach: Split the Hebrew text by sentence/period and distribute it 
        # back into the original blocks. This is an approximation but better than one large subtitle.
        
        # Split the translated text into chunks (sentences)
        # We use a non-greedy split based on common sentence delimiters (., !, ?)
        hebrew_chunks = re.split(r'([.!?])\s*', translated_full_text)
        # Filter out empty strings and re-combine the delimiter with the sentence
        hebrew_sentences = ["".join(i).strip() for i in zip(hebrew_chunks[0::2], hebrew_chunks[1::2])]
        # Add the last element if it was a final chunk without a delimiter
        if len(hebrew_chunks) % 2 != 0 and hebrew_chunks[-1].strip():
            hebrew_sentences.append(hebrew_chunks[-1].strip())
            
        translated_blocks = []
        sentence_index = 0
        
        for block in original_blocks:
            # If we have sentences left, use the next one as the subtitle for this block
            if sentence_index < len(hebrew_sentences):
                # Using the translated sentence as the text for the current block's timing
                block['text'] = hebrew_sentences[sentence_index]
                translated_blocks.append(block)
                sentence_index += 1
            else:
                 # If we run out of sentences, we just use the last one (shouldn't happen with good LLM output)
                 block['text'] = translated_blocks[-1]['text'] if translated_blocks else "..."
                 translated_blocks.append(block)

        # Generate the final SRT file content with Hebrew text and correct timings
        final_translated_srt = generate_new_srt(translated_blocks)
        
        return original_text, final_translated_srt
    
    except Exception as e:
         raise RuntimeError(f"Groq API call failed during transcription/translation/LLM. Error: {e}")
    
    finally:
        # Clean up the temporary audio file
        os.unlink(temp_audio_file_name)


# --- Telegram Handlers ---

@bot.message_handler(commands=['start'])
def send_welcome(message):
    """Handles the /start command and introduces the bot."""
    welcome_message = (
        "👋 שלום! אני הבוט של VidTransHeb.\n"
        "שלח לי סרטון (עד 5 דקות) ואני אשתקף ואתרגם את הכתוביות שלו לעברית, ואצרב אותן לוידאו.\n\n"
        "⏳ אנא המתן בסבלנות בזמן העיבוד, זה לוקח דקה או שתיים."
    )
    bot.send_message(message.chat.id, welcome_message)

@bot.message_handler(content_types=['video'])
def handle_video(message):
    """Handles incoming video files."""
    chat = message.chat.id
    
    # 1. Basic checks
    if message.video.duration > 300: # Check the 5 minute limit
        bot.send_message(chat, "❌ השגיאה: הסרטון ארוך מדי! אנא שלח סרטון של עד 5 דקות (300 שניות).")
        return

    # 2. Get the video file details
    file_info = bot.get_file(message.video.file_id)
    
    # Send acknowledgment immediately to avoid Telegram timeout
    ack_msg = bot.send_message(chat, "1/4. 📥 הסרטון התקבל. מפיק אודיו...")
    
    downloaded_file = bot.download_file(file_info.file_path)

    temp_paths = {} # Dictionary to store temp file paths for cleanup
    
    try:
        # --- Stage 1: Save Video to Temp File ---
        temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video_file.write(downloaded_file)
        temp_video_file.close()
        temp_paths['video'] = temp_video_file.name
        
        # --- Stage 2: Extract Audio from Video ---
        temp_audio_file = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
        temp_audio_file.close()
        temp_paths['audio'] = temp_audio_file.name
        
        # Use FFMPEG to extract audio stream
        (
            ffmpeg
            .input(temp_paths['video'])
            .output(temp_paths['audio'], acodec='libopus', b='64k') # Use opus for Groq compatibility
            .run(overwrite_output=True, quiet=True)
        )
        
        if os.path.getsize(temp_paths['audio']) < 1000:
             bot.edit_message_text("❌ לא הצלחתי להפיק אודיו מהסרטון, או שקובץ האודיו קטן מדי.", chat, ack_msg.message_id)
             return
             
        bot.edit_message_text("2/4. 🎤 אודיו הופק בהצלחה. מתחיל שעתוק, תרגום (LLM) וסנכרון...", chat, ack_msg.message_id)
        
        # --- Stage 3: Transcription and Translation (Whisper SRT + LLM) ---
        with open(temp_paths['audio'], "rb") as f:
            audio_bytes = f.read()
        
        # original_text is the transcript, final_translated_srt is the full SRT file content (Hebrew)
        original_text, final_translated_srt = get_transcript_and_translation(audio_bytes)

        if not final_translated_srt:
            bot.edit_message_text("❌ לא הצלחתי לזהות אודיו ברור או שלא נמצא טקסט לתרגום.", chat, ack_msg.message_id)
            return

        bot.edit_message_text("3/4. 📝 התרגום המסונכרן הושלם! מתחיל צריבת כתוביות לוידאו...", chat, ack_msg.message_id)

        # --- Stage 4: Create Subtitle File (SRT format) ---
        temp_sub_file = tempfile.NamedTemporaryFile(suffix=".srt", mode="w", encoding="utf-8", delete=False)
        temp_paths['sub'] = temp_sub_file.name
        
        # Write the final Hebrew SRT content
        temp_sub_file.write(final_translated_srt)
        temp_sub_file.close()

        # --- Stage 5: Burn Subtitles (FFMPEG) ---
        temp_output_file = tempfile.NamedTemporaryFile(suffix="_subbed.mp4", delete=False)
        temp_output_file.close()
        temp_paths['output'] = temp_output_file.name
        
        burn_subtitles_fast(temp_paths['video'], temp_paths['sub'], temp_paths['output'])

        bot.edit_message_text("4/4. 🎥 צריבת הכתוביות הסתיימה! שולח את הוידאו...", chat, ack_msg.message_id)

        # --- Stage 6: Send the Result ---
        caption_original = original_text[:150] + "..." if len(original_text) > 150 else original_text
        
        with open(temp_paths['output'], 'rb') as final_video:
            bot.send_video(
                chat, 
                final_video, 
                caption=f"✅ <b>סרטון מתורגם לעברית</b> באמצעות Groq.\n\n<u>הטקסט המקורי:</u> <i>{caption_original}</i>",
                supports_streaming=True
            )
        
        # Delete the acknowledgment message after completion
        bot.delete_message(chat, ack_msg.message_id)

    except Exception as e:
        error_type = type(e).__name__
        print(f"General Error: {error_type} - {e}")
        # Use traceback.format_exc() to get the full error details
        safe_send_error_message(
            chat, 
            f"אירעה שגיאה קריטית במהלך העיבוד ({error_type}).",
            traceback.format_exc()
        )
        
    finally:
        # --- Stage 7: Cleanup ---
        for path_type, path in temp_paths.items():
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception as e:
                    print(f"Error during cleanup of {path_type} file: {e}")

# --- Webhook and Server Setup ---

# Use environment variable for the webhook URL to be flexible (usually set by Render)
WEBHOOK_URL_BASE = os.environ.get('WEBHOOK_URL_BASE')
# The route Telegram will hit
WEBHOOK_URL_PATH = "/bot/" + BOT_TOKEN 

@app.route(WEBHOOK_URL_PATH, methods=['POST'])
def webhook():
    """Handles incoming webhook POST requests from Telegram."""
    if request.headers.get('content-type') == 'application/json':
        json_string = request.get_data().decode('utf-8')
        update = telebot.types.Update.de_json(json_string)
        bot.process_new_updates([update])
        return 'ok', 200
    return 'Content-Type must be application/json', 400

@app.route('/')
def home():
    """Simple Flask route for Render Heartbeat/Health check."""
    return {"status": "OK", "message": "Bot is running and listening for webhooks."}

def set_webhook_and_run():
    """Sets the Telegram Webhook URL and starts the Flask server."""
    if WEBHOOK_URL_BASE:
        full_url = WEBHOOK_URL_BASE.rstrip('/') + WEBHOOK_URL_PATH
        bot.remove_webhook()
        time.sleep(1) # Give Telegram a moment
        if bot.set_webhook(url=full_url):
             print(f"Webhook set successfully to: {full_url}")
        else:
             print("Failed to set webhook!")
    else:
        # Fallback to Polling if WEBHOOK_URL_BASE is not set (e.g., local run)
        print("WEBHOOK_URL_BASE not set. Starting TeleBot Polling...")
        # Start Polling in a thread if we are not using Webhooks
        threading.Thread(target=lambda: bot.polling(non_stop=True, interval=2), daemon=True).start()

    # Start the Flask web server (must be run in the main thread for Render)
    print("Flask Server מתחיל...")
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 10000))

if __name__ == '__main__':
    # We call the combined function to initialize webhook/polling and start the server
    set_webhook_and_run()
