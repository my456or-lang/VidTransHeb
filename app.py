import os
import threading
import tempfile
import traceback
import requests
from flask import Flask
import telebot
from groq import Groq
from deep_translator import GoogleTranslator
from moviepy.editor import VideoFileClip, CompositeVideoClip, ImageClip
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from bidi.algorithm import get_display
import arabic_reshaper
import time

# ---------------------------
#  CONFIG / ENV
# ---------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN לא מוגדר בסביבת הריצה")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY לא מוגדר בסביבת הריצה")

# ---------------------------
#  Clients
# ---------------------------
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)
client = Groq(api_key=GROQ_API_KEY)
translator = GoogleTranslator(source='auto', target='iw')

# ---------------------------
#  Flask (לשירות Render)
# ---------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Telegram subtitle bot — running ✅"


# ---------------------------
#  Prepare Hebrew text
# ---------------------------
def prepare_hebrew_text(text: str) -> str:
    try:
        reshaped = arabic_reshaper.reshape(text)
    except Exception:
        reshaped = text
    try:
        bidi_text = get_display(reshaped)
    except Exception:
        bidi_text = reshaped
    return bidi_text


# ---------------------------
# Load Hebrew Font
# ---------------------------
def find_font():
    custom_font = "fonts/NotoSansHebrew-VariableFont_wdth,wght.ttf"
    if os.path.exists(custom_font):
        return custom_font

    # fallback fonts
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    ]:
        if os.path.exists(p):
            return p

    return None


# ---------------------------
#  Transcription via Groq Whisper
# ---------------------------
def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f
        )
    if isinstance(resp, dict):
        return resp.get("text") or resp.get("transcript") or ""
    return getattr(resp, "text", "")


# ---------------------------
#  Create subtitle image (SAFE FOR PILLOW 10+)
# ---------------------------
def create_subtitle_image(text: str, video_w: int, video_h: int, fontsize: int = 56):
    text = prepare_hebrew_text(text)

    font_path = find_font()
    if font_path:
        font = ImageFont.truetype(font_path, fontsize)
    else:
        font = ImageFont.load_default()

    # Dummy image for measuring text
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)

    # Use textbbox — ONLY — no getsize()
    bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    padding = 25
    img = Image.new("RGBA", (text_w + padding*2, text_h + padding*2), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    # Background box
    d.rectangle([0, 0, img.width, img.height], fill=(0, 0, 0, 150))

    # Subtitle text
    d.text(
        (padding, padding),
        text,
        font=font,
        fill=(255, 255, 255, 255),
        stroke_width=2,
        stroke_fill=(0, 0, 0)
    )

    return img


# ---------------------------
# Burn subtitles onto video
# ---------------------------
def burn_subtitles_on_video(video_path: str, translated_text: str):
    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    subtitle_img = create_subtitle_image(translated_text, w, h, fontsize=max(28, int(w/30)))
    subtitle_np = np.array(subtitle_img)

    subtitle_clip = ImageClip(subtitle_np)\
        .set_duration(clip.duration)\
        .set_position(("center", h - subtitle_img.height - 40))

    final = CompositeVideoClip([clip, subtitle_clip])
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    final.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        threads=2,
        preset="ultrafast",
        verbose=False
    )

    clip.close()
    subtitle_clip.close()
    final.close()
    return out_path


# ---------------------------
#  Send progress message
# ---------------------------
def send_progress(chat_id, text):
    try:
        bot.send_message(chat_id, text)
    except:
        pass


# ---------------------------
#  Telegram message handler
# ---------------------------
@bot.message_handler(commands=['start'])
def on_start(msg):
    bot.reply_to(msg, "👋 היי! שלח סרטון (עד 5 דקות) ואחזיר לך אותו עם כתוביות בעברית.")


@bot.message_handler(content_types=['video'])
def handle_video(message):
    chat_id = message.chat.id

    try:
        send_progress(chat_id, "🎬 מוריד את הסרטון...")

        file_info = bot.get_file(message.video.file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
        resp = requests.get(file_url, timeout=120)

        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_video.write(resp.content)
        tmp_video.close()

        # Check length
        clip = VideoFileClip(tmp_video.name)
        if clip.duration > 5 * 60 + 5:
            bot.reply_to(message, "❌ הסרטון ארוך מדי — המקסימום הוא 5 דקות.")
            clip.close()
            return
        clip.close()

        send_progress(chat_id, "🎧 ממיר את האודיו לטקסט...")
        text = transcribe_audio(tmp_video.name)

        if not text.strip():
            bot.reply_to(message, "❌ לא זוהה דיבור בסרטון.")
            return

        send_progress(chat_id, "🔠 מתרגם לעברית...")
        try:
            translated = translator.translate(text)
        except:
            translated = text

        send_progress(chat_id, "🔥 שורף כתוביות על הווידאו...")
        out_video = burn_subtitles_on_video(tmp_video.name, translated)

        send_progress(chat_id, "📤 מעלה את הסרטון...")
        with open(out_video, "rb") as f:
            bot.send_video(chat_id, f, caption="✅ הנה הסרטון עם כתוביות!")

        os.remove(tmp_video.name)
        os.remove(out_video)

    except Exception as e:
        tb = traceback.format_exc()
        bot.reply_to(message, f"❌ שגיאה: {e}\n{tb}")


# ---------------------------
#  Run bot + Flask
# ---------------------------
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
