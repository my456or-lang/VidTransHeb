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
#  CONFIG / ENV
# ---------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
PORT = int(os.environ.get("PORT", 8080))

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN לא מוגדר בסביבת הריצה")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY לא מוגדר בסביבת הריצה")

# קבוע שנבחר: דחיית הכתוביות ביחס לזמנים שחוזרים מ־Whisper
OFFSET_SECONDS = 1.5  # כתוביות יופיעו מאוחר יותר ב־1.5 שניות

# ---------------------------
#  Clients
# ---------------------------
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)
client = Groq(api_key=GROQ_API_KEY)
translator = GoogleTranslator(source='auto', target='iw')  # deep-translator uses 'iw' עבור עברית

# ---------------------------
#  Flask (כדי ש־Render / שירותים אחרים ישימו לב לפורט)
# ---------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Telegram subtitle bot — running ✅"

# ---------------------------
#  Helpers: RTL + font
# ---------------------------
def prepare_hebrew_text_for_display(text: str) -> str:
    """
    For Hebrew: use bidi.get_display so text draws RTL.
    Avoid arabic_reshaper for Hebrew (it can break Hebrew).
    """
    try:
        return get_display(text)
    except Exception:
        return text

def find_font_path():
    # check repo fonts first (user-uploaded)
    custom = "fonts/NotoSansHebrew-VariableFont_wdth,wght.ttf"
    if os.path.exists(custom):
        return custom

    # common system fallbacks
    candidates = [
        "/usr/share/fonts/truetype/noto/NotoSansHebrew-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

# ---------------------------
#  Transcription via Groq Whisper (verbose_json -> segments)
# ---------------------------
def transcribe_with_groq_verbose(audio_file_path: str):
    """
    Calls Groq whisper with response_format=verbose_json to get segments.
    Returns list of segments: each an object/dict with keys: start, end, text
    """
    with open(audio_file_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="verbose_json"
        )
    # resp may be dict-like
    if isinstance(resp, dict):
        segments = resp.get("segments") or []
        # segments may be list of dicts with 'start','end','text'
        return segments
    # else if object-like
    return getattr(resp, "segments", []) or []

# ---------------------------
#  Image (PIL) subtitle creation (wrapped, right-aligned)
# ---------------------------
def wrap_text_lines(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, max_width: int):
    words = text.split()
    if not words:
        return ['']
    lines = []
    cur = words[0]
    for w in words[1:]:
        candidate = cur + ' ' + w
        bbox = draw.textbbox((0,0), candidate, font=font, stroke_width=2)
        if (bbox[2] - bbox[0]) <= max_width:
            cur = candidate
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines

def create_subtitle_image_block(text: str, video_w: int, fontsize: int = 48, max_width_ratio: float = 0.92):
    """
    Creates an RGBA image containing the subtitle block (right-aligned, RTL).
    Returns a PIL Image.
    """
    # prepare bidi for correct visual order
    display_text = prepare_hebrew_text_for_display(text)

    font_path = find_font_path()
    if font_path:
        font = ImageFont.truetype(font_path, fontsize)
    else:
        font = ImageFont.load_default()

    # dummy draw to measure
    dummy = Image.new("RGBA", (10,10), (0,0,0,0))
    draw = ImageDraw.Draw(dummy)

    max_text_w = int(video_w * max_width_ratio) - 40
    if max_text_w < 100:
        max_text_w = video_w - 40

    lines = wrap_text_lines(display_text, draw, font, max_text_w)

    # measure sizes
    line_sizes = [draw.textbbox((0,0), line, font=font, stroke_width=2) for line in lines]
    widths = [(b[2]-b[0]) for b in line_sizes]
    heights = [(b[3]-b[1]) for b in line_sizes]

    pad_x = 20
    pad_y = 12
    total_w = min(video_w - 40, max(widths) + pad_x*2)
    total_h = sum(heights) + pad_y * (len(lines) + 1)

    img = Image.new("RGBA", (int(total_w), int(total_h)), (0,0,0,0))
    d = ImageDraw.Draw(img)

    # background rectangle (black semi-opaque)
    d.rectangle([0,0,img.width,img.height], fill=(0,0,0,200))

    # draw lines, right-aligned
    y = pad_y
    for i, line in enumerate(lines):
        lw = widths[i]
        x = img.width - pad_x - lw  # right align
        # draw with stroke if supported
        try:
            d.text((x, y), line, font=font, fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0,255))
        except TypeError:
            # manual outline fallback
            outline = (0,0,0,255)
            for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
                d.text((x+ox, y+oy), line, font=font, fill=outline)
            d.text((x, y), line, font=font, fill=(255,255,255,255))
        y += heights[i] + pad_y

    return img

# ---------------------------
#  Burn timed subtitles (one clip per segment) with offset
# ---------------------------
def burn_timed_subtitles(video_path: str, segments: list, offset_seconds: float = OFFSET_SECONDS):
    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    subtitle_clips = []
    # create a single optimized font-size based on width
    base_fontsize = max(18, int(w / 36))

    for seg in segments:
        # seg may be dict or object
        start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", None)
        end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None)
        text = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")

        if start is None or end is None:
            continue
        # apply offset (positive -> delay; negative -> advance)
        s = max(0, float(start) + float(offset_seconds))
        e = max(s + 0.05, float(end) + float(offset_seconds))
        dur = e - s
        if dur <= 0:
            continue

        # create image for this segment (we reuse base fontsize, could adapt by length)
        img = create_subtitle_image_block(text, w, fontsize=base_fontsize)
        arr = np.array(img)
        subclip = ImageClip(arr).set_start(s).set_duration(dur).set_position(("center", h - img.height - 30))
        subtitle_clips.append(subclip)

    final = CompositeVideoClip([clip] + subtitle_clips)
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
    for sc in subtitle_clips:
        sc.close()
    final.close()
    return out_path

# ---------------------------
#  Helpers: messaging
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
    bot.reply_to(msg, "👋 היי! שלח סרטון (עד 5 דקות) ואני אחזיר אותו עם כתוביות בעברית (עם background שחור).")

@bot.message_handler(content_types=['video'])
def on_video(message):
    chat_id = message.chat.id
    try:
        send_progress(chat_id, "🎬 מוריד את הסרטון...")
        file_info = bot.get_file(message.video.file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
        resp = requests.get(file_url, timeout=120)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(resp.content)
        tmp.close()

        # quick duration check
        clip = VideoFileClip(tmp.name)
        if clip.duration > 5 * 60 + 5:
            bot.reply_to(message, "❌ הסרטון ארוך מדי — המקסימום הוא 5 דקות.")
            clip.close()
            os.remove(tmp.name)
            return
        clip.close()

        send_progress(chat_id, "🎧 מתמלל (Whisper)...")
        try:
            segments = transcribe_with_groq_verbose(tmp.name)
        except Exception as e:
            tb = traceback.format_exc()
            bot.reply_to(message, f"❌ שגיאה בתמלול: {e}\n{tb}")
            os.remove(tmp.name)
            return

        if not segments:
            bot.reply_to(message, "❌ לא נמצאו קטעים בתמלול.")
            os.remove(tmp.name)
            return

        send_progress(chat_id, "🔠 מתרגם כל קטע לעברית...")
        # translate per segment (robust)
        translated_segments = []
        for seg in segments:
            text = seg.get("text") if isinstance(seg, dict) else getattr(seg, "text", "")
            try:
                heb = translator.translate(text) if text and text.strip() else ""
            except Exception:
                heb = text
            # keep start/end as floats
            start = seg.get("start") if isinstance(seg, dict) else getattr(seg, "start", None)
            end = seg.get("end") if isinstance(seg, dict) else getattr(seg, "end", None)
            if start is None or end is None:
                continue
            translated_segments.append({"start": float(start), "end": float(end), "text": heb})

        if not translated_segments:
            bot.reply_to(message, "❌ לא נוצרו כתוביות מתורגמות.")
            os.remove(tmp.name)
            return

        send_progress(chat_id, "🔥 רינדור הכתוביות בסרטון (זה עשוי לקחת כמה דקות)...")
        try:
            out_video = burn_timed_subtitles(tmp.name, translated_segments, offset_seconds=OFFSET_SECONDS)
        except Exception as e:
            tb = traceback.format_exc()
            bot.reply_to(message, f"❌ שגיאה ביצירת הכתוביות: {e}\n{tb}")
            os.remove(tmp.name)
            return

        send_progress(chat_id, "📤 מעלה את הסרטון המתורגם...")
        with open(out_video, "rb") as fh:
            bot.send_video(chat_id, fh, caption="✅ הנה הסרטון שלך עם כתוביות בעברית.")

        # cleanup
        try:
            os.remove(tmp.name)
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
#  Run bot + Flask for Render
# ---------------------------
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=PORT)
