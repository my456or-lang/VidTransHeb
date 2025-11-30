import os
import threading
import tempfile
import traceback
import requests
from flask import Flask
import telebot
from groq import Groq
from deep_translator import GoogleTranslator
# מהייבוא של moviepy/PIL נשאר רק מה שדרוש לטעינת הגופן
from PIL import ImageFont
import ffmpeg # **חדש: נדרש לחילוץ אודיו וצריבה מהירה**

# ============================================
# ENV
# ============================================
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN לא מוגדר")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY לא מוגדר")

bot = telebot.TeleBot(BOT_TOKEN)
client = Groq(api_key=GROQ_API_KEY)
translator = GoogleTranslator(source="auto", target="iw")

app = Flask(__name__)

@app.route("/")
def home():
    return "Telegram Hebrew Subtitle Bot — Running ✅"


# ============================================
# FONT
# ============================================
def get_hebrew_font(size=48):
    """מחזיר נתיב לגופן העברי."""
    font_path = "fonts/NotoSansHebrew.ttf"
    if os.path.exists(font_path):
        # במקרה של הרצה מקומית
        return font_path 
    # עבור סביבות דוקר/רנדר שבהן הנתיב מוגדר ב-Dockerfile
    return "NotoSansHebrew.ttf" 


# ============================================
# CREATE SRT FILE (משתמש בלוגיקת ה-segments הקיימת)
# ============================================
def create_srt_file(segments, offset=1.8):
    """יוצר קובץ SRT מקטעי התרגום."""
    srt_path = tempfile.NamedTemporaryFile(delete=False, suffix=".srt", mode="w", encoding="utf-8").name
    
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments):
            start = seg["start"] + offset
            end = seg["end"] + offset
            text = seg["text"]

            # פורמט זמן SRT: HH:MM:SS,mmm
            start_str = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02},{int((start * 1000) % 1000):03}"
            end_str = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02},{int((end * 1000) % 1000):03}"

            f.write(f"{i + 1}\n")
            f.write(f"{start_str} --> {end_str}\n")
            f.write(f"{text}\n\n")

    return srt_path


# ============================================
# BURN SUBTITLES (FFMPEG DIRECT - **מהיר**)
# ============================================
def burn_subtitles_fast(video_path, srt_path):
    
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    font_name = os.path.basename(get_hebrew_font()) # קבלת השם 'NotoSansHebrew.ttf'

    # הגדרות צריבה מלאות ל-libass (הכלי ש-FFMPEG משתמש בו לכתוביות מתקדמות)
    # Alignment=2 (תחתון מרכז), Outline=2, PrimaryColour לבן.
    # Fontname= משתמש בשם הקובץ שהועתק לתוך images/fontsdir ב-Dockerfile
    style = f"Fontname={font_name}:PrimaryColour=&H00FFFFFF:OutlineColour=&H00000000:Outline=2:Shadow=0:Spacing=1.5:BorderStyle=3:Alignment=2"

    # FFMPEG מבצע את הצריבה והקידוד מחדש בבת אחת - הכי מהיר
    try:
        (
            ffmpeg
            .input(video_path)
            .output(out_path, 
                    vf=f"subtitles={srt_path}:force_style='{style}'",
                    vcodec='libx264',
                    acodec='aac',
                    preset='ultrafast', # **המפתח למהירות: קידוד מהיר**
                    crf=23 # איכות פלט טובה
            )
            .run(overwrite_output=True, quiet=True) # quiet=True מפחית פלט לקונסולה
        )
    except ffmpeg.Error as e:
        # טיפול בשגיאות ffmpeg
        error_message = e.stderr.decode('utf8')
        raise RuntimeError(f"שגיאה בצריבת כתוביות (FFMPEG): {error_message}")

    return out_path


# ============================================
# TELEGRAM HANDLER
# ============================================
def send_progress(chat_id, text):
    """שולח הודעה למשתמש, עם טיפול בשגיאות."""
    try:
        bot.send_message(chat_id, text)
    except:
        pass


@bot.message_handler(commands=["start"])
def start(msg):
    bot.reply_to(msg, "🎬 שלח סרטון עד 5 דקות ואחזיר אותו עם כתוביות בעברית — מסונכרנות!")


@bot.message_handler(content_types=["video"])
def handle_video(message):
    chat = message.chat.id
    temp_video_path = None
    temp_audio_path = None
    temp_srt_path = None
    final_output_path = None

    try:
        # --- 1. הורדה ובדיקת וידאו ---
        send_progress(chat, "📥 מוריד את הסרטון...")
        file_info = bot.get_file(message.video.file_id)
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
        data = requests.get(url).content

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(data)
        temp_file.close()
        temp_video_path = temp_file.name

        # בדיקת אורך סרטון באמצעות FFMPEG (יותר מהיר מ-moviepy)
        probe = ffmpeg.probe(temp_video_path)
        duration = float(probe['format']['duration'])
        if duration > 305:
            bot.send_message(chat, "❌ הסרטון ארוך מ־5 דקות.")
            return

        # --- 2. חילוץ אודיו (מהיר!) ושליחה ל-Groq ---
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
        
        send_progress(chat, "🎶 מפיק אודיו ושולח ל-Groq API...")

        # חילוץ אודיו באמצעות FFMPEG (מהיר)
        (
            ffmpeg
            .input(temp_video_path)
            .output(temp_audio_path, vn=None, acodec='libmp3lame')
            .run(overwrite_output=True, quiet=True)
        )
        
        # זיהוי דיבור ותרגום (רק של קובץ האודיו הקטן)
        with open(temp_audio_path, "rb") as audio_file:
            resp = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=audio_file, 
                response_format="verbose_json"
            )
        
        segments = resp.segments

        # --- 3. תרגום ויצירת SRT ---
        send_progress(chat, "🌍 מתרגם כל שורה...")
        for s in segments:
            s["text"] = translator.translate(s["text"])
        
        temp_srt_path = create_srt_file(segments, offset=1.8) 

        # --- 4. צריבת כתוביות (FFMPEG מהיר!) ושליחה ---
        send_progress(chat, "🔥 שורף כתוביות (FFMPEG מהיר!)...")
        final_output_path = burn_subtitles_fast(temp_video_path, temp_srt_path) 

        send_progress(chat, "📤 מעלה את הסרטון...")
        with open(final_output_path, "rb") as f:
            bot.send_video(chat, f, caption="✅ הנה הסרטון שלך!")

    except Exception as e:
        bot.send_message(chat, f"❌ שגיאה: {e}\n{traceback.format_exc()}")
    
    finally:
        # --- 5. ניקוי קבצים זמניים ---
        files_to_remove = [temp_video_path, temp_audio_path, temp_srt_path, final_output_path]
        for f in files_to_remove:
            if f and os.path.exists(f):
                os.remove(f)


# ============================================
# RUN
# ============================================
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    threading.Thread(target=run_bot, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
