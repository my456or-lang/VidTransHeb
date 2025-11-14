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
translator = GoogleTranslator(source='auto', target='he')

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
    """
    Apply reshaping (for Arabic-like scripts) and bidi display.
    For Hebrew get_display is important to render right-to-left.
    """
    try:
        # arabic_reshaper won't break hebrew; kept to handle mixed RTL scripts too
        reshaped = arabic_reshaper.reshape(text)
    except Exception:
        reshaped = text
    try:
        bidi_text = get_display(reshaped)
    except Exception:
        bidi_text = reshaped
    return bidi_text

def find_font_path():
    # preferred fonts that usually include Hebrew glyphs
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ---------------------------
#  Transcription via Groq Whisper
# ---------------------------
def transcribe_audio(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            # use whisper model
            resp = client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f
            )
        # Depending on Groq response shape, try to extract text robustly
        if isinstance(resp, dict):
            # older style
            return resp.get("text") or resp.get("transcript") or ""
        # object with attribute
        return getattr(resp, "text", "") or getattr(resp, "transcript", "") or str(resp)
    except Exception as e:
        # return empty and raise later
        raise

# ---------------------------
#  Create subtitle image (style B: white text with black stroke, bottom)
# ---------------------------
def create_subtitle_image(text: str, video_w: int, video_h: int, fontsize: int = 56, max_width_ratio: float = 0.9):
    """
    Returns a PIL Image (RGBA) containing the subtitle block to overlay.
    Style B: white text with black stroke, aligned to right, placed near bottom.
    """
    # Prepare text for RTL
    text = prepare_hebrew_text(text)

    # Choose font
    font_path = find_font_path()
    if not font_path:
        # fallback to default PIL font (smaller, might not have Hebrew)
        font = ImageFont.load_default()
    else:
        font = ImageFont.truetype(font_path, fontsize)

    max_width = int(video_w * max_width_ratio)
    # We will wrap text to lines that fit max_width
    # simple greedy wrap using ImageDraw.textbbox
    dummy_img = Image.new("RGBA", (10, 10), (0,0,0,0))
    draw = ImageDraw.Draw(dummy_img)

    words = text.split()
    lines = []
    current = ""
    for w in words:
        candidate = (w + " " + current).strip()  # keep RTL-friendly order for drawing
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

    # Now compute image height
    line_height = font.getsize("A")[1] + 12 if hasattr(font, "getsize") else fontsize + 12
    total_height = line_height * len(lines) + 30  # padding

    img = Image.new("RGBA", (video_w, total_height), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    # draw semi-opaque black rounded rectangle background (subtle)
    padding_x = 20
    padding_y = 10
    rect_left = (video_w - max_width) // 2 - padding_x
    rect_right = rect_left + max_width + padding_x*2
    rect_top = 0
    rect_bottom = total_height
    # semi-transparent dark rectangle
    draw.rectangle([rect_left, rect_top, rect_right, rect_bottom], fill=(0,0,0,180))

    # draw lines from top->bottom but each line right-aligned
    y = padding_y
    for line in lines:
        # compute bbox for this line
        bbox = draw.textbbox((0,0), line, font=font, stroke_width=2)
        tw = bbox[2] - bbox[0]
        x = rect_right - padding_x - tw  # right align inside rect
        # Pillow supports stroke_* params for text since recent versions
        try:
            draw.text((x, y), line, font=font, fill=(255,255,255,255),
                      stroke_width=2, stroke_fill=(0,0,0,255), align="right")
        except TypeError:
            # older Pillow compatibility: draw outline manually
            outline_color = (0,0,0,255)
            for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
                draw.text((x+ox, y+oy), line, font=font, fill=outline_color)
            draw.text((x, y), line, font=font, fill=(255,255,255,255))
        y += line_height

    return img

# ---------------------------
#  Burn subtitles onto video (hard-coded)
# ---------------------------
def burn_subtitles_on_video(video_path: str, translated_text: str) -> str:
    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    # create subtitle image sized to video width
    subtitle_img = create_subtitle_image(translated_text, w, h, fontsize=max(28, int(w/30)))
    # Convert to ImageClip
    subtitle_np = np.array(subtitle_img)
    subtitle_clip = ImageClip(subtitle_np).set_duration(clip.duration).set_position(("center", h - subtitle_img.height - 20))

    final = CompositeVideoClip([clip, subtitle_clip])
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    # write with reasonable settings
    final.write_videofile(out_path, codec="libx264", audio_codec="aac", threads=2, preset="ultrafast", verbose=False)
    clip.close()
    subtitle_clip.close()
    final.close()
    return out_path

# ---------------------------
#  small helper: send progress messages
# ---------------------------
def send_progress(chat_id, text):
    try:
        bot.send_message(chat_id, text)
    except Exception:
        pass

# ---------------------------
#  Telegram handler
# ---------------------------
@bot.message_handler(commands=['start'])
def on_start(msg):
    bot.reply_to(msg, "👋 היי! שלח סרטון (עד 5 דקות) ואחזיר לך אותו עם כתוביות בעברית (Hard-coded).")

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
            bot.reply_to(message, f"❌ שגיאה בתרגום: {e}")
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
            bot.send_video(chat_id, f, caption="✅ הנה הסרטון שלך עם כתוביות בעברית (Hard-coded).")

        # cleanup
        try:
            os.remove(tmp_video.name)
        except OSError:
            pass
        try:
            os.remove(out_video)
        except OSError:
            pass

    except Exception as e:
        tb = traceback.format_exc()
        bot.reply_to(message, f"❌ שגיאה כללית: {e}\n{tb}")

# ---------------------------
#  Run bot + Flask for Render
# ---------------------------
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
