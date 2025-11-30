import os
import io
import time
import tempfile
import traceback
import threading
from dotenv import load_dotenv

# Import for Telegram and Flask
import telebot
from flask import Flask
from telebot import apihelper

# Import for FFMPEG (Video processing) and Groq (Transcription)
import ffmpeg
from groq import Groq

# Load environment variables (used for local testing, Render uses its own Environment variables)
load_dotenv()

# --- Configuration ---
BOT_TOKEN = os.environ.get('TELEGRAM_TOKEN')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
ADMIN_USER_ID = os.environ.get('ADMIN_USER_ID') # Optional: For admin alerts
FFMPEG_TIMEOUT = 300 # 5 minutes timeout for FFMPEG (No longer used directly in run())

# Initialize Clients
try:
    bot = telebot.TeleBot(BOT_TOKEN)
    groq_client = Groq(api_key=GROQ_API_KEY)
except ValueError as e:
    # This will now catch the 'Token must not contain spaces' error during Flask startup
    print(f"FATAL ERROR: Failed to initialize TeleBot. Check TELEGRAM_TOKEN for spaces. Error: {e}")
    exit(1)
except Exception as e:
    print(f"FATAL ERROR: Failed to initialize clients. Check environment variables. Error: {e}")
    exit(1)

# Initialize Flask App
app = Flask(__name__)

# --- Helper Functions ---

# Function to safely limit the error message length for Telegram's 4096 character limit
def safe_send_error_message(chat_id, error_message, full_traceback=""):
    """
    Sends an error message to the user, ensuring the text does not exceed 4096 characters.
    """
    full_message = f"❌ שגיאה: {error_message}\n\n"
    
    # Check if we have a detailed traceback to add
    if full_traceback:
        # FFMPEG tracebacks can be very long. We truncate the full traceback to ensure
        # the total message length does not exceed Telegram's limit (4096).
        # We reserve 500 characters for the initial error message and cut the traceback.
        MAX_TRACEBACK_LEN = 3500 
        
        if len(full_traceback) > MAX_TRACEBACK_LEN:
            full_traceback = full_traceback[:MAX_TRACEBACK_LEN] + "\n... [המשך השגיאה קוצץ] ..."
        
        full_message += f"פרטים טכניים:\n```\n{full_traceback}\n```"
    
    # Send the final (and now safe) message
    try:
        bot.send_message(chat_id, full_message, parse_mode='Markdown')
    except apihelper.ApiTelegramException as e:
        # If even the safe message fails, send the most basic message
        bot.send_message(chat_id, f"❌ שגיאה קריטית: לא ניתן לשלוח את פרטי השגיאה. (שגיאה: {e})")

# FFMPEG Command: Subtitle burning and re-encoding
def burn_subtitles_fast(input_path, subtitle_path, output_path):
    """
    Uses FFMPEG to burn subtitles into the video file using the 'subtitles' filter.
    """
    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                # Subtitles filter: using simple font parameters for maximum compatibility
                # Fontname='Noto Sans Hebrew' is crucial for Hebrew support
                vf=f"subtitles='{subtitle_path}':force_style='Fontname=Noto Sans Hebrew,FontSize=28,Alignment=10,Outline=2,Shadow=1,MarginV=40'",
                vcodec='libx264',
                acodec='copy',
                pix_fmt='yuv420p',
                preset='ultrafast',  # Fast encoding speed
                crf=23, # Default quality for H.264
                strict='experimental' # Allows non-standard features if needed
            )
            .global_args('-t', '300') # Hard limit of 5 minutes (300 seconds)
            # FIX: Removed 'timeout=FFMPEG_TIMEOUT' as it causes TypeError in the Render environment.
            .run(overwrite_output=True, quiet=True, capture_stdout=True, capture_stderr=True)
        )
        return True
    except ffmpeg.Error as e:
        # Pass the FFMPEG error details back to the main handler
        raise RuntimeError(f"FFMPEG Encoding Failed. Stderr: {e.stderr.decode('utf8', errors='ignore')}")
    except Exception as e:
        raise e

# Groq Transcription and Translation (Using Whisper only)
def get_transcript_and_translation(audio_data):
    """
    Transcribes audio to English and translates the audio directly to Hebrew
    using two separate calls to the Whisper model (to avoid complex SRT parsing).
    """
    # Create a temporary file to hold the audio data
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as temp_audio_file:
        temp_audio_file.write(audio_data)
        temp_audio_file_name = temp_audio_file.name

    try:
        # 1. Transcribe the audio file to ENGLISH (to get the original text for the caption)
        with open(temp_audio_file_name, "rb") as audio_file:
            transcript_response = groq_client.audio.transcriptions.create(
                file=(temp_audio_file_name, audio_file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en" # Source language assumption
            )
            original_text = transcript_response.text

        if not original_text:
            return None, "לא נמצא טקסט לשעתוק."

        # 2. Translate the audio file directly to HEBREW using Whisper
        # We must reopen the file stream for the second API call
        with open(temp_audio_file_name, "rb") as audio_file:
            translation_response = groq_client.audio.translations.create(
                file=(temp_audio_file_name, audio_file.read()),
                model="whisper-large-v3",
                response_format="json",
                # The prompt guides Whisper to output the translation in the target language (Hebrew)
                prompt="Translate the content into formal Hebrew." 
            )
            # Whisper translation returns the translated text directly
            translated_text = translation_response.text
        
        return original_text, translated_text
    
    # Catching Groq API errors specifically, although they should be avoided now.
    except Exception as e:
         # Changed to RuntimeError to be consistent with FFMPEG error handling
         raise RuntimeError(f"Groq/Whisper API call failed during transcription/translation. Error: {e}")
    
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
    downloaded_file = bot.download_file(file_info.file_path)

    temp_paths = {} # Dictionary to store temp file paths for cleanup
    
    try:
        # --- Stage 1: Save Video to Temp File ---
        temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_video_file.write(downloaded_file)
        temp_video_file.close()
        temp_paths['video'] = temp_video_file.name
        
        bot.send_message(chat, "1/4. 📥 הסרטון התקבל ונשמר באופן זמני. מפיק אודיו...")

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
        
        # We must ensure the audio file is correctly created and has content
        if os.path.getsize(temp_paths['audio']) < 1000:
             bot.send_message(chat, "❌ לא הצלחתי להפיק אודיו מהסרטון, או שקובץ האודיו קטן מדי.")
             return
             
        bot.send_message(chat, "2/4. 🎤 אודיו הופק בהצלחה. מתחיל שעתוק ותרגום (Whisper)...")
        
        # --- Stage 3: Transcription and Translation (Whisper Only) ---
        with open(temp_paths['audio'], "rb") as f:
            audio_bytes = f.read()
        
        # Calling the updated function
        original_text, translated_text = get_transcript_and_translation(audio_bytes)

        if not translated_text:
            bot.send_message(chat, "❌ לא הצלחתי לזהות אודיו ברור או שלא נמצא טקסט לתרגום.")
            return

        bot.send_message(chat, "3/4. 📝 התרגום הושלם! מתחיל צריבת כתוביות לוידאו...")

        # --- Stage 4: Create Subtitle File (SRT format) ---
        # Whisper translation returns the full translated script. We use a single SRT block.
        
        temp_sub_file = tempfile.NamedTemporaryFile(suffix=".srt", mode="w", encoding="utf-8", delete=False)
        temp_paths['sub'] = temp_sub_file.name
        
        # Simple SRT structure: Full translated text displayed for the entire video duration
        temp_sub_file.write("1\n")
        # Start and end time (from 0 seconds to video duration)
        duration_srt = time.strftime('%H:%M:%S,000', time.gmtime(message.video.duration))
        temp_sub_file.write(f"00:00:00,000 --> {duration_srt}\n")
        temp_sub_file.write(translated_text + "\n")
        temp_sub_file.close()

        # --- Stage 5: Burn Subtitles (FFMPEG) ---
        temp_output_file = tempfile.NamedTemporaryFile(suffix="_subbed.mp4", delete=False)
        temp_output_file.close()
        temp_paths['output'] = temp_output_file.name
        
        # Call the fixed FFMPEG function (now without the unsupported 'timeout' argument)
        burn_subtitles_fast(temp_paths['video'], temp_paths['sub'], temp_paths['output'])

        bot.send_message(chat, "4/4. 🎥 צריבת הכתוביות הסתיימה! שולח את הוידאו...")

        # --- Stage 6: Send the Result ---
        # Shorten original text for caption
        caption_original = original_text[:100] + "..." if len(original_text) > 100 else original_text
        
        with open(temp_paths['output'], 'rb') as final_video:
            bot.send_video(
                chat, 
                final_video, 
                caption=f"✅ סרטון מתורגם לעברית באמצעות Groq Whisper.\n\nהטקסט המקורי: {caption_original}",
                supports_streaming=True
            )

    except Exception as e:
        # Send a safe, truncated error message
        error_type = type(e).__name__
        print(f"General Error: {error_type} - {e}")
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
                    print(f"Cleaned up temporary file: {path}")
                except Exception as e:
                    print(f"Error during cleanup of {path_type} file: {e}")


@app.route('/')
def home():
    """Simple Flask route for Render Heartbeat/Health check."""
    return {"status": "OK", "message": "Bot is running in Polling mode."}

# --- Main Execution (The Threading Fix) ---

def run_bot_polling():
    """Starts the TeleBot Polling in a separate thread."""
    print("TeleBot Polling מתחיל...")
    # Use non_stop=True to auto-reconnect if connection drops
    bot.polling(non_stop=True, interval=2) 

if __name__ == '__main__':
    # Start the bot polling loop in a background thread
    threading.Thread(target=run_bot_polling, daemon=True).start()
    
    # Start the Flask web server (must be run in the main thread for Render)
    print("Flask Server מתחיל...")
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 10000))
