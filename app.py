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

# optional: if python-bidi is installed we can use it to ensure RTL display
try:
    from bidi.algorithm import get_display
except Exception:
    get_display = None

# ---------------------------
# CONFIG / ENV
# ---------------------------
BOT_TOKEN = os.environ.get("TELEGRAM_TOKEN")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("TELEGRAM_TOKEN לא מוגדר")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY לא מוגדר")

# clients
bot = telebot.TeleBot(BOT_TOKEN, parse_mode=None)
client = Groq(api_key=GROQ_API_KEY)
# deep-translator mapping for Hebrew uses 'iw' in some versions; 'he' may also work.
translator = GoogleTranslator(source='auto', target='iw')

app = Flask(__name__)

@app.route("/")
def home():
    return "Telegram subtitle bot — running ✅"

# ---------------------------
# Fonts
# ---------------------------
def find_font_path():
    # preferred bundled font
    local_font = "fonts/NotoSansHebrew.ttf"
    if os.path.exists(local_font):
        return local_font
    # try alternative common paths
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def get_font(size):
    path = find_font_path()
    if path:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    # fallback
    return ImageFont.load_default()

# ---------------------------
# Small helper: prepare text for display (RTL)
# ---------------------------
def prepare_display_text(text: str) -> str:
    # If bidi available, use it for display ordering. Do NOT apply arabic_reshaper to Hebrew.
    if get_display:
        try:
            return get_display(text)
        except Exception:
            return text
    return text

# ---------------------------
# Wrapping text into lines using draw.textbbox (safe for Pillow >= 10)
# ---------------------------
def wrap_text(text: str, font: ImageFont.ImageFont, max_width: int):
    # simple greedy wrap on spaces
    words = text.split()
    if not words:
        return [""]
    lines = []
    draw_dummy = ImageDraw.Draw(Image.new("RGBA", (10,10)))
    current = words[0]
    for w in words[1:]:
        candidate = current + " " + w
        bbox = draw_dummy.textbbox((0,0), candidate, font=font, stroke_width=2)
        wbox = bbox[2] - bbox[0]
        if wbox <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines

# ---------------------------
# Create subtitle image for a given text (right-aligned inside the box)
# returns a PIL RGBA image
# ---------------------------
def create_subtitle_image(text: str, video_w: int, fontsize: int = 48, max_width_ratio: float = 0.92):
    # prepare display ordering for RTL languages
    display_text = prepare_display_text(text)

    # select font
    font = get_font(fontsize)

    # dummy draw to measure
    draw_dummy = ImageDraw.Draw(Image.new("RGBA", (10,10)))
    max_text_width = int(video_w * max_width_ratio) - 40
    if max_text_width < 100:
        max_text_width = video_w - 40

    lines = wrap_text(display_text, font, max_text_width)

    # measure lines
    linewidths = []
    lineheights = []
    for line in lines:
        bbox = draw_dummy.textbbox((0,0), line, font=font, stroke_width=2)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        linewidths.append(w)
        lineheights.append(h)

    padding_x = 24
    padding_y = 10
    total_w = min(video_w - 40, max(linewidths) + padding_x*2)
    total_h = sum(lineheights) + padding_y*(len(lines)+1)

    # create RGBA image
    img = Image.new("RGBA", (int(total_w), int(total_h)), (0,0,0,0))
    draw = ImageDraw.Draw(img)

    # semi-transparent background rectangle (fill entire image)
    draw.rectangle([0,0,img.width,img.height], fill=(0,0,0,160))

    # draw lines right-aligned inside the rectangle
    y = padding_y
    for i, line in enumerate(lines):
        lw = linewidths[i]
        x = img.width - padding_x - lw  # right alignment
        # attempt to draw with stroke (outline)
        try:
            draw.text((x, y), line, font=font, fill=(255,255,255,255),
                      stroke_width=2, stroke_fill=(0,0,0,255))
        except TypeError:
            # older pillow fallback - manual outline
            for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
                draw.text((x+ox, y+oy), line, font=font, fill=(0,0,0,255))
            draw.text((x, y), line, font=font, fill=(255,255,255,255))
        y += lineheights[i] + padding_y

    return img

# ---------------------------
# Transcribe audio via Groq (verbose_json to get segments)
# ---------------------------
def transcribe_with_segments(audio_path: str):
    with open(audio_path, "rb") as f:
        resp = client.audio.transcriptions.create(
            model="whisper-large-v3-turbo",
            file=f,
            response_format="verbose_json"  # IMPORTANT: returns segments with timing
        )
    # resp might be dict-like or object
    if isinstance(resp, dict):
        segments = resp.get("segments", [])
    else:
        segments = getattr(resp, "segments", None)
        if segments is None:
            # try to dict-ify
            try:
                d = resp.__dict__
                segments = d.get("segments", [])
            except Exception:
                segments = []
    # Normalize segments: ensure list of dicts with start,end,text
    out = []
    for s in segments:
        try:
            start = s.get("start", s.get("t_start", None) if isinstance(s, dict) else getattr(s, "start", None))
            end = s.get("end", s.get("t_end", None) if isinstance(s, dict) else getattr(s, "end", None))
            text = s.get("text", None) if isinstance(s, dict) else getattr(s, "text", "")
            if text is None:
                text = ""
            out.append({"start": float(start), "end": float(end), "text": text.strip()})
        except Exception:
            continue
    return out

# ---------------------------
# Burn timed subtitles: create a clip per segment
# ---------------------------
def burn_timed_subtitles(video_path: str, translated_segments):
    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    subtitle_clips = []
    for seg in translated_segments:
        seg_text = seg["text"]
        seg_start = seg["start"]
        seg_end = seg["end"]
        duration = max(0.1, seg_end - seg_start)

        # create image for this segment
        fontsize = max(20, int(w / 36))
        img = create_subtitle_image(seg_text, w, fontsize=fontsize)
        arr = np.array(img)

        # create ImageClip
        ic = ImageClip(arr).set_start(seg_start).set_duration(duration).set_position(("center", h - img.height - 30))
        subtitle_clips.append(ic)

    # composite
    final = CompositeVideoClip([clip] + subtitle_clips)
    out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

    # write file
    final.write_videofile(
        out_path,
        codec="libx264",
        audio_codec="aac",
        preset="ultrafast",
        threads=2,
        verbose=False
    )

    clip.close()
    for c in subtitle_clips:
        try:
            c.close()
        except:
            pass
    final.close()
    return out_path

# ---------------------------
# Telegram handlers
# ---------------------------
def send_progress(chat_id, text):
    try:
        bot.send_message(chat_id, text)
    except Exception:
        pass

@bot.message_handler(commands=['start'])
def on_start(msg):
    bot.reply_to(msg, "👋 היי! שלח סרטון עד 5 דקות ואחזיר אותו עם כתוביות מסונכרנות לעברית.")

@bot.message_handler(content_types=['video'])
def handle_video(message):
    chat_id = message.chat.id
    try:
        send_progress(chat_id, "📥 מוריד את הסרטון...")
        file_info = bot.get_file(message.video.file_id)
        file_url = f"https://api.telegram.org/file/bot{BOT_TOKEN}/{file_info.file_path}"
        r = requests.get(file_url, timeout=180)
        if r.status_code != 200:
            bot.reply_to(message, "❌ שגיאה בהורדה מהשרת של טלגרם.")
            return

        tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_video.write(r.content)
        tmp_video.close()

        # quick duration check
        clip = VideoFileClip(tmp_video.name)
        if clip.duration > 5 * 60 + 5:
            clip.close()
            bot.reply_to(message, "❌ הסרטון ארוך מדי — המקסימום 5 דקות.")
            os.remove(tmp_video.name)
            return
        clip.close()

        send_progress(chat_id, "🎧 מתמלל ומייצר זמני כתוביות (מבוסס Groq Whisper)...")
        segments = transcribe_with_segments(tmp_video.name)
        if not segments:
            bot.reply_to(message, "❌ לא נוצרו קטעי תמלול (segments).")
            os.remove(tmp_video.name)
            return

        # translate each segment
        send_progress(chat_id, "🔠 מתרגם כל קטע לעברית...")
        translated_segments = []
        for s in segments:
            txt = s.get("text", "").strip()
            if not txt:
                continue
            try:
                heb = translator.translate(txt)
            except Exception:
                heb = txt
            # prepare display ordering
            heb_display = prepare_display_text(heb)
            translated_segments.append({"start": s["start"], "end": s["end"], "text": heb_display})

        if not translated_segments:
            bot.reply_to(message, "❌ לאחר התרגום לא נותר טקסט לתצוגה.")
            os.remove(tmp_video.name)
            return

        send_progress(chat_id, "🔥 שורף כתוביות לפי זמנים על הווידאו (עשוי לקחת כמה דקות)...")
        out_path = burn_timed_subtitles(tmp_video.name, translated_segments)

        send_progress(chat_id, "📤 מעלה את הווידאו בחזרה...")
        with open(out_path, "rb") as f:
            bot.send_video(chat_id, f, caption="✅ הנה הסרטון עם כתוביות מסונכרנות לעברית")

        # cleanup
        try:
            os.remove(tmp_video.name)
        except:
            pass
        try:
            os.remove(out_path)
        except:
            pass

    except Exception as e:
        tb = traceback.format_exc()
        bot.reply_to(message, f"❌ שגיאה כללית: {e}\n{tb}")

# ---------------------------
# Run bot + Flask (for Render)
# ---------------------------
def run_bot():
    bot.infinity_polling(timeout=60, long_polling_timeout=60)

if __name__ == "__main__":
    t = threading.Thread(target=run_bot, daemon=True)
    t.start()
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
