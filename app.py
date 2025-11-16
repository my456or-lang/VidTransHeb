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
    font_path = "fonts/NotoSansHebrew.ttf"
    if os.path.exists(font_path):
        return ImageFont.truetype(font_path, size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)


# ============================================
# WRAP RTL TEXT (WITHOUT bidi, WITHOUT reshape!)
# ============================================
def wrap_rtl(text, draw, font, max_width):
    words = text.split()
    lines = []
    current = ""

    for w in words:
        test = w if not current else current + " " + w
        w_box = draw.textbbox((0,0), test, font=font, stroke_width=2)[2]

        if w_box <= max_width:
            current = test
        else:
            lines.append(current)
            current = w

    if current:
        lines.append(current)
    return lines


# ============================================
# SUBTITLE IMAGE
# ============================================
def create_subtitle_image(text, video_w, video_h):
    fontsize = max(24, int(video_w / 34))
    font = get_hebrew_font(fontsize)

    dummy = Image.new("RGBA", (10,10), (0,0,0,0))
    draw = ImageDraw.Draw(dummy)

    max_width = int(video_w * 0.90)
    lines = wrap_rtl(text, draw, font, max_width)

    sizes = [draw.textbbox((0,0), line, font=font, stroke_width=2) for line in lines]
    widths = [(x2-x1) for (x1,y1,x2,y2) in sizes]
    heights = [(y2-y1) for (x1,y1,x2,y2) in sizes]

    pad_x = 25
    pad_y = 12

    total_w = min(video_w - 40, max(widths) + pad_x * 2)
    total_h = sum(heights) + pad_y*(len(lines)+1)

    img = Image.new("RGBA", (total_w, total_h), (0,0,0,160))
    draw2 = ImageDraw.Draw(img)

    y = pad_y
    for i, line in enumerate(lines):
        lw = widths[i]
        x = total_w - pad_x - lw  # RIGHT ALIGN

        draw2.text(
            (x, y),
            line,
            font=font,
            fill=(255,255,255,255),
            stroke_width=2,
            stroke_fill=(0,0,0,255)
        )
        y += heights[i] + pad_y

    return img


# ============================================
# BURN SUBTITLES (SEGMENTED)
# ============================================
def burn_subtitles(video_path, segments, offset=1.0):

    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    subtitle_clips = []

    for seg in segments:
        start = seg["start"] + offset
        end   = seg["end"] + offset
        text  = seg["text"]

        img = create_subtitle_image(text, w, h)
        img_np = np.array(img)

        sub = (
            ImageClip(img_np)
            .set_start(max(0, start))
            .set_duration(max(0.05, end - start))
            .set_position(("center", h - img.height - 30))
        )

        subtitle_clips.append(sub)

    final = CompositeVideoClip([clip] + subtitle_clips)

    out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    final.write_videofile(
        out,
        codec="libx264",
        audio_codec="aac",
        threads=2,
        preset="ultrafast",
        verbose=False
    )

    clip.close()
    final.close()
    return out


# ============================================
# TELEGRAM
# ============================================
def send_progress(chat_id, text):
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

    try:
        send_progress(chat, "📥 מוריד את הסרטון...")
        file_info = bot.get_file(message.video.file_id)
        url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
        data = requests.get(url).content

        temp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp.write(data)
        temp.close()

        clip = VideoFileClip(temp.name)
        if clip.duration > 305:
            bot.send_message(chat, "❌ הסרטון ארוך מ־5 דקות.")
            return
        clip.close()

        send_progress(chat, "🎧 מפענח אודיו (כולל זמנים)...")

        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=open(temp.name, "rb"),
            response_format="verbose_json"
        )

        segments = resp.segments

        # Translate each segment
        send_progress(chat, "🌍 מתרגם...")
        for s in segments:
            s["text"] = translator.translate(s["text"])

        send_progress(chat, "🔥 שורף כתוביות (עם offset של 1.8 שנייה)...")
        out_path = burn_subtitles(temp.name, segments, offset=1.0)

        send_progress(chat, "📤 מעלה את הסרטון...")
        with open(out_path, "rb") as f:
            bot.send_video(chat, f, caption="✅ הנה הסרטון שלך!")

        os.remove(temp.name)
        os.remove(out_path)

    except Exception as e:
        bot.send_message(chat, f"❌ שגיאה: {e}\n{traceback.format_exc()}")


# ============================================
# RUN
# ============================================
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)


if __name__ == "__main__":
    threading.Thread(target=run_bot, daemon=True).start()
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
