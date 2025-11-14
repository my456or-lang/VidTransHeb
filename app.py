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
import time

BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN לא מוגדר")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY לא מוגדר")

bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)
client = Groq(api_key=GROQ_API_KEY)
translator = GoogleTranslator(source='auto', target='iw')

app = Flask(__name__)

@app.route("/")
def home():
    return "Telegram subtitle bot — running ✅"

# ---------- FONT ----------
def get_hebrew_font(fontsize=48):
    local_font = "fonts/NotoSansHebrew.ttf"
    if os.path.exists(local_font):
        return ImageFont.truetype(local_font, fontsize)

    # fallback
    return ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        fontsize
    )

# ---------- WRAP TEXT RTL ----------
def wrap_text(text, draw, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for w in words:
        test = (w if not current else current + " " + w)
        bbox = draw.textbbox((0, 0), test, font=font, stroke_width=2)
        w_box = bbox[2] - bbox[0]

        if w_box <= max_width:
            current = test
        else:
            lines.append(current)
            current = w

    if current:
        lines.append(current)

    return lines

# ---------- CREATE SUBTITLE RECTANGLE ----------
def create_subtitle_image(text, video_w, video_h):

    fontsize = max(22, int(video_w / 32))
    font = get_hebrew_font(fontsize)

    dummy = Image.new("RGBA", (10, 10), (0,0,0,0))
    draw = ImageDraw.Draw(dummy)

    max_text_width = int(video_w * 0.92)

    lines = wrap_text(text, draw, font, max_text_width)

    # Measure rectangle size
    line_sizes = [draw.textbbox((0,0), line, font=font, stroke_width=2) for line in lines]
    widths = [(x2-x1) for (x1,y1,x2,y2) in line_sizes]
    heights = [(y2-y1) for (x1,y1,x2,y2) in line_sizes]

    padding_x = 25
    padding_y = 12

    total_w = min(video_w - 40, max(widths) + padding_x * 2)
    total_h = sum(heights) + padding_y * (len(lines)+1)

    img = Image.new("RGBA", (total_w, total_h), (0,0,0,160))
    draw2 = ImageDraw.Draw(img)

    y = padding_y
    for i, line in enumerate(lines):
        line_width = widths[i]
        x = total_w - padding_x - line_width  # RIGHT ALIGN

        draw2.text(
            (x, y), line,
            font=font,
            fill=(255,255,255,255),
            stroke_width=2,
            stroke_fill=(0,0,0,255)
        )

        y += heights[i] + padding_y

    return img

# ---------- BURN SUBTITLES ----------
def burn_subtitles_on_video(video_path, translated_text):

    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    img = create_subtitle_image(translated_text, w, h)
    img_np = np.array(img)

    subtitle = ImageClip(img_np)\
        .set_duration(clip.duration)\
        .set_position(("center", h - img.height - 30))

    final = CompositeVideoClip([clip, subtitle])

    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    final.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=2,
        verbose=False
    )

    clip.close()
    subtitle.close()
    final.close()
    return out_path

# ---------- TELEGRAM ----------
def send_progress(chat_id, text):
    try:
        bot.send_message(chat_id, text)
    except:
        pass

@bot.message_handler(commands=['start'])
def on_start(msg):
    bot.reply_to(msg, "👋 שלח סרטון עד 5 דקות ואחזיר אותו עם כתוביות בעברית!")

@bot.message_handler(content_types=['video'])
def handle_video(message):

    chat_id = message.chat.id

    try:
        send_progress(chat_id, "📥 מוריד את הסרטון...")
        file_info = bot.get_file(message.video.file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
        video_bytes = requests.get(file_url).content

        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_video.write(video_bytes)
        temp_video.close()

        clip = VideoFileClip(temp_video.name)
        if clip.duration > 305:
            bot.reply_to(message, "❌ הסרטון ארוך מדי (מעל 5 דקות)")
            clip.close()
            return
        clip.close()

        send_progress(chat_id, "🎧 מפענח אודיו...")
        text = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=open(temp_video.name, "rb")
        ).text

        send_progress(chat_id, "🌍 מתרגם לעברית...")
        translated = translator.translate(text)

        send_progress(chat_id, "🔥 שורף כתוביות על הווידאו...")
        out_path = burn_subtitles_on_video(temp_video.name, translated)

        send_progress(chat_id, "📤 מעלה את הסרטון...")
        with open(out_path, "rb") as f:
            bot.send_video(chat_id, f, caption="הנה הסרטון עם כתוביות! ✅")

        os.remove(temp_video.name)
        os.remove(out_path)

    except Exception as e:
        bot.send_message(chat_id, f"❌ שגיאה: {e}\n{traceback.format_exc()}")

# ---------- START ----------
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
