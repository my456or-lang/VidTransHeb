import os
import io
import time
import tempfile
import traceback
import threading
import re
import json # Import for JSON parsing
import html # NEW: Import for HTML escaping the traceback
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
    The traceback is HTML-escaped to prevent Telegram's parser from misinterpreting code snippets 
    (like <listcomp>) as invalid HTML tags.
    """
    full_message = f"❌ <b>שגיאה קריטית:</b> {error_message}\n\n"
    
    if full_traceback:
        MAX_TRACEBACK_LEN = 3000
        
        if len(full_traceback) > MAX_TRACEBACK_LEN:
            full_traceback = full_traceback[:MAX_TRACEBACK_LEN] + "\n... [המשך השגיאה קוצץ] ..."
        
        # CRITICAL FIX: Escape HTML characters in the traceback to prevent Telegram from misinterpreting them.
        escaped_traceback = html.escape(full_traceback)
        
        full_message += f"<u>פרטים טכניים:</u>\n<pre>{escaped_traceback}</pre>"
    
    try:
        # Use HTML parse mode for better formatting and code block support
        bot.send_message(chat_id, full_message, parse_mode='HTML')
    except telebot.apihelper.ApiTelegramException as e:
        # Fallback if even the error sending fails (e.g., due to extreme length or other API issues)
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

def convert_seconds_to_srt_time(seconds):
    """Converts a floating-point number of seconds to SRT time format: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining_seconds = seconds % 60
    milliseconds = int((remaining_seconds - int(remaining_seconds)) * 1000)
    
    return f"{hours:02}:{minutes:02}:{int(remaining_seconds):02},{milliseconds:03}"

def generate_new_srt(translated_segments):
    """
    Generates a new SRT string from the translated segments.
    Each segment contains 'start', 'end' (in seconds), and 'text' (Hebrew).
    """
    new_srt = ""
    for i, segment in enumerate(translated_segments):
        start_time = convert_seconds_to_srt_time(segment['start'])
        end_time = convert_seconds_to_srt_time(segment['end'])
        
        new_srt += f"{i + 1}\n"
        new_srt += f"{start_time} --> {end_time}\n"
        new_srt += f"{segment['text']}\n\n"
        
    return new_srt.strip()

def get_transcript_and_translation(audio_data):
    """
    1. Transcribes audio to verbose_json (to get time synchronization).
    2. Uses Groq LLM to translate the full transcript to Hebrew.
    3. Rebuilds the segments with the Hebrew translation.
    Returns: original_text (str), translated_srt_content (str)
    """
    # Create a temporary file to hold the audio data
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_name = temp_audio_file.name

    try:
        # --- 1. Transcription to Verbose JSON (for sync) ---
        # FIX: Changed response_format to 'verbose_json' which is supported and provides segments/timestamps
        with open(temp_audio_file_name, "rb") as audio_file:
            transcript_response_json = groq_client.audio.transcriptions.create(
                file=(temp_audio_file_name, audio_file.read()),
                model="whisper-large-v3",
                response_format="verbose_json", 
                language="en" # Source language
            )
            # transcript_response_json is a ChatCompletion object, we need to extract the JSON string if needed,
            # but usually the Python client handles the deserialization for us, which we check below.
        
        # Check if the response is a dictionary/object with the expected structure
        if not hasattr(transcript_response_json, 'segments') or not transcript_response_json.segments:
             raise RuntimeError("Whisper did not return valid segments for synchronization.")
             
        original_segments = transcript_response_json.segments
        original_text = transcript_response_json.text
        
        # --- 2. Translation using Groq LLM ---
        
        # System instruction to guide the LLM's output
        system_prompt = "You are a professional subtitle translator. Your task is to translate a large block of text that has been segmented into subtitle-length chunks. Translate the following list of segments into high-quality, clear, and colloquial Hebrew. The output MUST be a valid JSON array, where each element is a string containing the Hebrew translation for the corresponding English segment. The output MUST ONLY contain the JSON array, nothing else."

        # Create a list of the original English segment texts for the LLM
        original_segment_texts = [s.text.strip() for s in original_segments]
        
        # Use the JSON structure to ensure the LLM returns the output in the required format
        # This is CRITICAL for matching the original segments back to the translations.
        
        user_query = f"Translate the following array of English segments into Hebrew. Provide the output as a JSON array of strings:\n\n{json.dumps(original_segment_texts, ensure_ascii=False)}"
        
        translation_response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            model=LLM_MODEL,
            temperature=0.1
        )
        
        # Extract the JSON string from the LLM response
        translated_json_string = translation_response.choices[0].message.content.strip()

        # Clean up the JSON string (LLMs sometimes add ```json and extra text)
        translated_json_string = re.sub(r"```json|```", "", translated_json_string, flags=re.IGNORECASE).strip()
        
        # Parse the JSON array
        translated_texts = json.loads(translated_json_string)
        
        if len(translated_texts) != len(original_segments):
             raise RuntimeError(f"Translation segments mismatch: Expected {len(original_segments)} translations but got {len(translated_texts)}.")

        # --- 3. Rebuild the Segments with Hebrew Text ---
        
        final_segments = []
        for i, original_segment in enumerate(original_segments):
            final_segments.append({
                'start': original_segment.start, # Time in seconds (float)
                'end': original_segment.end,   # Time in seconds (float)
                'text': translated_texts[i]    # Translated Hebrew text (str)
            })

        # Generate the final SRT file content with Hebrew text and correct timings
        final_translated_srt = generate_new_srt(final_segments)
        
        return original_text, final_translated_srt
    
    except json.JSONDecodeError as e:
        # Include a snippet of the raw output for debugging JSON issues
        raw_output_snippet = translated_json_string[:500] if 'translated_json_string' in locals() else "N/A"
        raise RuntimeError(f"LLM translation failed to return clean JSON output for parsing. Error: {e} - Raw LLM Output: {raw_output_snippet}...")
    except Exception as e:
         # Catch Groq's BadRequestError and re-raise it nicely
         raise RuntimeError(f"Groq API call failed during processing. Error: {e}")
    
    finally:
        # Clean up the temporary audio file
        if 'temp_audio_file_name' in locals() and os.path.exists(temp_audio_file_name):
            os.unlink(temp_audio_file_name)


# --- Telegram Handlers ---
# (The Telegram, Webhook, and FFMPEG parts remain the same as Fix 7)

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
        
        # --- Stage 3: Transcription and Translation (Whisper Verbose JSON + LLM) ---
        with open(temp_paths['audio'], "rb") as f:
            audio_bytes = f.read()
        
        # original_text is the full transcript, final_translated_srt is the full SRT file content (Hebrew, synchronized)
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
