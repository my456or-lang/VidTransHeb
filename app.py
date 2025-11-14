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
#  Flask (כדי ש-Render יראה פורט פתוח)
# ---------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Telegram subtitle bot — running ✅"


# ---------------------------
#  Utilities: RTL + text shaping
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
#  Hebrew Font Loader (your font)
# ---------------------------
def find_font_path():
    """
    Load your Hebrew font from the bundled fonts folder.
    """
    local_font = "fonts/NotoSansHebrew-VariableFont_wdth,wght.ttf"
    if os.path.exists(local_font):
        return local_font
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
    return getattr(resp, "text", "") or getattr(resp, "transcript", "") or str(resp)


# ---------------------------
#  Create subtitle image
# ---------------------------
def create_subtitle_image(text: str, video_w: int, video_h: int, fontsize: int = 56, max_width_ratio: float = 0.9):
    text = prepare_hebrew_text(text)

    font_path = find_font_path()
    if not font_path:
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, fontsize)

    max_width = int(video_w * max_width_ratio)

    dummy_img = Image.new("RGBA", (10, 10), (0,0,0,0))
    draw = ImageDraw.Draw(dummy_img)

    words = text.split()
    lines = []
    current = ""
    for w in words:
        candidate = (w + " " + current).strip()
        bbox = draw.textbbox((0,0), candidate, font=font)
        wbox = bbox[2] - bbox[0]
        if wbox <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)

    line_height = font.getsize("A")[1] + 12
    total_height = line_height * len(lines) + 30

    img = Image.new("RGBA", (video_w, total_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    padding_x = 20
    padding_y = 10
    rect_left = (video_w - max_width) // 2 - padding_x
    rect_right = rect_left + max_width + padding_x*2
    rect_top = 0
    rect_bottom = total_height

    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=(0,0,0,180))

    y = padding_y
    for line in lines:
        bbox = draw.textbbox((0,0), line, font=font, stroke_width=2)
        tw = bbox[2] - bbox[0]
        x = rect_right - padding_x - tw
        draw.text((x, y), line, font=font, fill=(255,255,255,255),
                  stroke_width=2, stroke_fill=(0,0,0,255))
        y += line_height

    return img


# ---------------------------
#  Burn subtitles
# ---------------------------
def burn_subtitles_on_video(video_path: str, translated_text: str) -> str:
    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    subtitle_img = create_subtitle_image(translated_text, w, h, fontsize=max(28, int(w/30)))

    subtitle_np = np.array(subtitle_img)
    subtitle_clip = ImageClip(subtitle_np).set_duration(clip.duration).set_position(("center", h - subtitle_img.height - 20))

    final = CompositeVideoClip([clip, subtitle_clip])
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    final.write_videofile(out_path, codec="libx264", audio_codec="aac", threads=2, preset="ultrafast", verbose=False)

    clip.close()
    subtitle_clip.close()
    final.close()

    return out_path


# ---------------------------
#  Send progress
# ---------------------------
def send_progress(chat_id, text):
    try:
        bot.send_message(chat_id, text)
    except Exception:
        pass


# ---------------------------
#  Telegram handlers
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
        tmp_video.flush()
        tmp_video.close()

        try:
            clip = VideoFileClip(tmp_video.name)
            dur = clip.duration
            clip.close()
        except:
            dur = None

        if dur and dur > 305:
            bot.reply_to(message, "❌ הסרטון ארוך מדי — המקסימום הוא 5 דקות.")
            os.remove(tmp_video.name)
            return

        send_progress(chat_id, "🎧 ממיר את האודיו לטקסט...")

        try:
            text = transcribe_audio(tmp_video.name)
        except Exception as e:
            bot.reply_to(message, f"❌ שגיאה בתמלול: {e}")
            os.remove(tmp_video.name)
            return

        if not text.strip():
            bot.reply_to(message, "❌ לא זוהה דיבור בסרטון.")
            os.remove(tmp_video.name)
            return

        send_progress(chat_id, "🔠 מתרגם לעברית...")

        try:
            translated = translator.translate(text)
        except Exception:
            translated = text

        send_progress(chat_id, "🔥 שורף כתוביות על הווידאו...")

        try:
            out_video = burn_subtitles_on_video(tmp_video.name, translated)
        except Exception as e:
            bot.reply_to(message, f"❌ שגיאה ביצירת כתוביות:\n{e}")
            os.remove(tmp_video.name)
            return

        send_progress(chat_id, "📤 מעלה את הסרטון...")

        with open(out_video, "rb") as f:
            bot.send_video(chat_id, f, caption="✅ הנה הסרטון שלך עם כתוביות בעברית!")

        os.remove(tmp_video.name)
        os.remove(out_video)

    except Exception as e:
        bot.reply_to(message, f"❌ שגיאה כללית: {e}")


# ---------------------------
#  Run
# ---------------------------
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
