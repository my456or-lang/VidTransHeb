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
import time

# ==========================================================
#  CONFIG
# ==========================================================
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN לא מוגדר")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY לא מוגדר")

bot = telebot.TeleBot(BOT_TOKEN)
client = Groq(api_key=GROQ_API_KEY)
translator = GoogleTranslator(source='auto', target='iw')

# ==========================================================
#  FLASK (Render keepalive)
# ==========================================================
app = Flask(__name__)

@app.route("/")
def home():
    return "Subtitle bot running ✔️"


# ==========================================================
#  Prepare RTL Hebrew text
# ==========================================================
def prepare_hebrew_text(text: str) -> str:
    """Fix RTL order using bidi only – NO arabic reshaping."""
    try:
        return get_display(text)
    except:
        return text


# ==========================================================
#  Load font
# ==========================================================
def find_font():
    custom_font = "fonts/NotoSansHebrew-VariableFont_wdth,wght.ttf"
    if os.path.exists(custom_font):
        return custom_font

    # fallback fonts
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(p):
            return p

    return None


# ==========================================================
#  Wrap text into RTL lines
# ==========================================================
def wrap_text(text, draw, font, max_width):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]

    for w in words[1:]:
        test = current + " " + w
        bbox = draw.textbbox((0, 0), test, font=font, stroke_width=2)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = test
        else:
            lines.append(current)
            current = w

    lines.append(current)
    return lines


# ==========================================================
#  Transcribe audio
# ==========================================================
def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f
        )
    return getattr(resp, "text", "") or ""


# ==========================================================
#  Create subtitle image (multi-line RTL)
# ==========================================================
def create_subtitle_image(text: str, video_w: int):
    text = prepare_hebrew_text(text)
    font_path = find_font()
    font = ImageFont.truetype(font_path, 48) if font_path else ImageFont.load_default()

    dummy = Image.new("RGBA", (10, 10))
    draw = ImageDraw.Draw(dummy)

    max_width = int(video_w * 0.88)
    lines = wrap_text(text, draw, font, max_width)

    # measure
    line_heights = []
    line_widths = []

    for ln in lines:
        bbox = draw.textbbox((0, 0), ln, font=font, stroke_width=2)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_widths.append(w)
        line_heights.append(h)

    pad_x = 24
    pad_y = 10

    img_w = max(line_widths) + pad_x * 2
    img_h = sum(line_heights) + pad_y * (len(lines) + 1)

    img = Image.new("RGBA", (img_w, img_h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img_w, img_h], fill=(0, 0, 0, 160))

    y = pad_y
    for i, ln in enumerate(lines):
        line_w = line_widths[i]
        x = img_w - pad_x - line_w  # RTL right align
        d.text((x, y), ln, font=font,
               fill="white", stroke_width=2, stroke_fill="black")
        y += line_heights[i] + pad_y

    return img


# ==========================================================
#  Burn onto video
# ==========================================================
def burn_subtitles_on_video(video_path: str, text: str):
    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    img = create_subtitle_image(text, w)
    arr = np.array(img)

    sub = ImageClip(arr).set_duration(clip.duration)
    sub = sub.set_position(("center", h - img.height - 40))

    final = CompositeVideoClip([clip, sub])
    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    final.write_videofile(out, codec="libx264", audio_codec="aac",
                          threads=2, preset="ultrafast", verbose=False)

    clip.close()
    sub.close()
    final.close()

    return out


# ==========================================================
#  Telegram handlers
# ==========================================================
def send_progress(chat_id, msg):
    try:
        bot.send_message(chat_id, msg)
    except:
        pass


@bot.message_handler(commands=['start'])
def start(msg):
    bot.reply_to(msg, "🎬 שלח סרטון (עד 5 דקות) ואחזיר עם כתוביות בעברית.")


@bot.message_handler(content_types=['video'])
def handle_video(message):
    chat_id = message.chat.id

    try:
        send_progress(chat_id, "📥 מוריד את הסרטון...")

        file_info = bot.get_file(message.video.file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"

        resp = requests.get(file_url, timeout=120)

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(resp.content)
        tmp.close()

        # length check
        clip = VideoFileClip(tmp.name)
        if clip.duration > 305:
            bot.send_message(chat_id, "❌ עד 5 דקות בלבד.")
            clip.close()
            return
        clip.close()

        send_progress(chat_id, "🔊 מפענח אודיו...")
        text = transcribe_audio(tmp.name)

        if not text.strip():
            bot.send_message(chat_id, "❌ לא זוהה דיבור.")
            return

        send_progress(chat_id, "🌍 מתרגם לעברית...")
        try:
            heb = translator.translate(text)
        except:
            heb = text

        send_progress(chat_id, "🔥 שורף כתוביות על הווידאו...")
        out = burn_subtitles_on_video(tmp.name, heb)

        send_progress(chat_id, "📤 מעלה את הסרטון...")
        with open(out, "rb") as f:
            bot.send_video(chat_id, f, caption="✔️ מוכן!")

        os.remove(tmp.name)
        os.remove(out)

    except Exception as e:
        bot.send_message(chat_id, f"⚠️ שגיאה: {e}")
        print(traceback.format_exc())


# ==========================================================
#  RUN
# ==========================================================
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
