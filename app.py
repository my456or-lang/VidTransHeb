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

# ---------------------------
# CONFIG / ENV
# ---------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN לא מוגדר בסביבת הריצה")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY לא מוגדר בסביבת הריצה")

# ---------------------------
# CLIENTS
# ---------------------------
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)
client = Groq(api_key=GROQ_API_KEY)

# deep_translator מצפה לשמות כמו 'hebrew' -> ממפה ל-'iw'
translator = GoogleTranslator(source='auto', target='hebrew')

# ---------------------------
# Flask (so render sees an open port)
# ---------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Telegram subtitle bot — running ✅"

# ---------------------------
# Utils: font, find font path
# ---------------------------
def find_font_path():
    # try common Linux fonts that contain Hebrew glyphs
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ---------------------------
# Transcription via Groq Whisper (robust extraction)
# ---------------------------
def transcribe_audio(file_path: str) -> str:
    with open(file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f
        )
    # resp could be dict-like or object with .text
    if isinstance(resp, dict):
        return resp.get("text") or resp.get("transcript") or ""
    return getattr(resp, "text", "") or getattr(resp, "transcript", "") or str(resp)

# ---------------------------
# Wrapping and RTL handling
# ---------------------------
def wrap_text_rtl_logical(text: str, draw: ImageDraw.Draw, font, max_width: int):
    """
    Build lines for display. We keep logical order when building lines
    (so words are concatenated in their logical order), but measure
    visual width using get_display() to ensure correct wrapping for RTL.
    Returns list of lines (logical strings).
    """
    words = text.strip().split()
    if not words:
        return []

    lines = []
    current = words[0]

    for w in words[1:]:
        candidate_logical = current + " " + w  # keep logical order
        candidate_visual = get_display(candidate_logical)
        bbox = draw.textbbox((0,0), candidate_visual, font=font, stroke_width=2)
        width = bbox[2] - bbox[0]
        if width <= max_width:
            current = candidate_logical
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines

# ---------------------------
# Create subtitle image (RTL-ready)
# ---------------------------
def create_subtitle_image(text: str, video_w: int, fontsize: int = 48, max_width_ratio: float = 0.9):
    """
    Returns PIL RGBA image containing subtitle block (bottom-centered),
    with right alignment and stroke for readability.
    Input 'text' is logical Hebrew text (normal).
    """
    font_path = find_font_path()
    if font_path:
        font = ImageFont.truetype(font_path, fontsize)
    else:
        font = ImageFont.load_default()

    max_width = int(video_w * max_width_ratio)

    # dummy image for measuring
    dummy = Image.new("RGBA", (10,10), (0,0,0,0))
    draw = ImageDraw.Draw(dummy)

    lines_logical = wrap_text_rtl_logical(text, draw, font, max_width)

    # compute line height
    try:
        ascent, descent = font.getmetrics()
        line_h = ascent + descent + 8
    except Exception:
        line_h = fontsize + 12

    total_height = line_h * len(lines_logical) + 20  # padding

    img = Image.new("RGBA", (video_w, total_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    # background rectangle
    padding_x = 18
    padding_y = 8
    rect_w = max_width + padding_x * 2
    rect_left = (video_w - rect_w) // 2
    rect_right = rect_left + rect_w
    rect_top = 0
    rect_bottom = total_height
    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=(0,0,0,160))

    # draw each line right-aligned inside rect
    y = padding_y
    for logical_line in lines_logical:
        visual_line = get_display(logical_line)  # prepare visual order for rendering
        bbox = draw.textbbox((0,0), visual_line, font=font, stroke_width=2)
        tw = bbox[2] - bbox[0]
        x = rect_right - padding_x - tw  # right align
        # draw with stroke if available
        try:
            draw.text((x, y), visual_line, font=font, fill=(255,255,255,255),
                      stroke_width=2, stroke_fill=(0,0,0,255))
        except TypeError:
            # older Pillow - emulate stroke
            outline = (0,0,0,255)
            for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
                draw.text((x+ox, y+oy), visual_line, font=font, fill=outline)
            draw.text((x, y), visual_line, font=font, fill=(255,255,255,255))
        y += line_h

    return img

# ---------------------------
# Burn subtitles onto video
# ---------------------------
def burn_subtitles_on_video(video_path: str, translated_text: str) -> str:
    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    fontsize = max(26, int(w / 28))
    subtitle_img = create_subtitle_image(translated_text, video_w=w, fontsize=fontsize)
    subtitle_np = np.array(subtitle_img)
    subtitle_clip = ImageClip(subtitle_np).set_duration(clip.duration).set_position(("center", h - subtitle_img.height - 20))

    final = CompositeVideoClip([clip, subtitle_clip])
    out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    # use ultrafast preset to speed up on small instances; adjust as needed
    final.write_videofile(out_tmp, codec="libx264", audio_codec="aac", threads=2, preset="ultrafast", verbose=False)
    clip.close()
    subtitle_clip.close()
    final.close()
    return out_tmp

# ---------------------------
# small helper: send progress messages
# ---------------------------
def send_progress(chat_id, text):
    try:
        bot.send_message(chat_id, text)
    except Exception:
        pass

# ---------------------------
# Telegram handlers
# ---------------------------
@bot.message_handler(commands=['start'])
def on_start(msg):
    bot.reply_to(msg, "👋 היי! שלח סרטון (עד 5 דקות) ואחזיר לך אותו עם כתוביות בעברית בתחתית הסרטון.")

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

        # quick duration check
        try:
            clip = VideoFileClip(tmp_video.name)
            dur = clip.duration
            clip.close()
        except Exception:
            dur = None

        if dur and dur > 5 * 60 + 5:
            bot.reply_to(message, "❌ הסרטון ארוך מדי — המקסימום הוא 5 דקות.")
            os.remove(tmp_video.name)
            return

        send_progress(chat_id, "🎧 ממיר את האודיו לטקסט (תמלול)...")
        try:
            text = transcribe_audio(tmp_video.name)
        except Exception as e:
            bot.reply_to(message, f"❌ שגיאה בתמלול: {e}")
            os.remove(tmp_video.name)
            return

        if not text or len(text.strip()) == 0:
            bot.reply_to(message, "❌ לא זוהה דיבור בסרטון.")
            os.remove(tmp_video.name)
            return

        send_progress(chat_id, "🔠 מתרגם לעברית...")
        try:
            translated = translator.translate(text)
        except Exception as e:
            # אם יש בעיה בתרגום נחזיר את הטקסט המקורי (באנגלית) ונמשיך
            bot.reply_to(message, f"⚠ שגיאה בתרגום — ישלח הטקסט המקורי.\n{e}")
            translated = text

        send_progress(chat_id, "🔥 שורף כתוביות על הווידאו (זה עשוי לקחת כמה דקות)...")
        try:
            out_video = burn_subtitles_on_video(tmp_video.name, translated)
        except Exception as e:
            tb = traceback.format_exc()
            bot.reply_to(message, f"❌ קרתה שגיאה ביצירת הכתוביות:\n{e}\n{tb}")
            os.remove(tmp_video.name)
            return

        send_progress(chat_id, "📤 מעלה את הסרטון בחזרה...")
        with open(out_video, "rb") as f:
            bot.send_video(chat_id, f, caption="✅ הנה הסרטון שלך עם כתוביות בעברית.")

        # cleanup
        try:
            os.remove(tmp_video.name)
        except Exception:
            pass
        try:
            os.remove(out_video)
        except Exception:
            pass

    except Exception as e:
        tb = traceback.format_exc()
        bot.reply_to(message, f"❌ שגיאה כללית: {e}\n{tb}")

# ---------------------------
# Run bot + Flask for Render
# ---------------------------
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
