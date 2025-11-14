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
# note: we keep arabic_reshaper installed for cases with Arabic text, but we won't force-apply it to Hebrew
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
translator = GoogleTranslator(source='auto', target='iw')  # 'iw' is deep-translator mapping for Hebrew

# ---------------------------
#  Flask (לשירות Render)
# ---------------------------
app = Flask(__name__)

@app.route("/")
def home():
    return "Telegram subtitle bot — running ✅"

# ---------------------------
#  RTL helper: use bidi.get_display
#  DON'T forcibly reshape Hebrew using arabic_reshaper (it can break Hebrew).
#  We only attempt to reshape if the text likely contains Arabic-range characters.
# ---------------------------
def prepare_hebrew_text(text: str) -> str:
    try:
        # if contains arabic letters (basic heuristic) try reshape; otherwise skip
        if any('\u0600' <= ch <= '\u06FF' for ch in text):
            try:
                reshaped = arabic_reshaper.reshape(text)
            except Exception:
                reshaped = text
        else:
            reshaped = text
    except Exception:
        reshaped = text

    try:
        bidi_text = get_display(reshaped)
    except Exception:
        bidi_text = reshaped
    return bidi_text

# ---------------------------
# Load Hebrew font from repo (user added fonts/...)
# ---------------------------
def find_font():
    # prefer bundled NotoHebrew if present
    custom_font = "fonts/NotoSansHebrew-VariableFont_wdth,wght.ttf"
    if os.path.exists(custom_font):
        return custom_font

    # fallback fonts commonly available in Debian images
    for p in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf"
    ]:
        if os.path.exists(p):
            return p

    return None

# ---------------------------
# Wrap text into lines to fit max_width using draw.textbbox
# ---------------------------
def wrap_text_to_lines(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, max_width: int):
    words = text.split()
    if not words:
        return ['']

    lines = []
    current = words[0]
    for w in words[1:]:
        candidate = current + ' ' + w
        bbox = draw.textbbox((0,0), candidate, font=font, stroke_width=2)
        wbox = bbox[2] - bbox[0]
        if wbox <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = w
    lines.append(current)
    return lines

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
    return getattr(resp, "text", "") or str(resp)

# ---------------------------
#  Create subtitle image with wrapping and right alignment
# ---------------------------
def create_subtitle_image(text: str, video_w: int, video_h: int, fontsize: int = 48, max_width_ratio: float = 0.92):
    # prepare RTL text (bidi)
    text = prepare_hebrew_text(text)

    font_path = find_font()
    if font_path:
        font = ImageFont.truetype(font_path, fontsize)
    else:
        font = ImageFont.load_default()

    # dummy draw for measuring
    dummy = Image.new("RGBA", (10, 10), (0,0,0,0))
    draw = ImageDraw.Draw(dummy)

    max_text_width = int(video_w * max_width_ratio) - 40  # reserve padding
    # if max_text_width is small keep sane minimum
    if max_text_width < 100:
        max_text_width = video_w - 40

    # wrap into lines
    lines = wrap_text_to_lines(text, draw, font, max_text_width)

    # measure each line height and width
    line_heights = []
    line_widths = []
    for line in lines:
        bbox = draw.textbbox((0,0), line, font=font, stroke_width=2)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        line_widths.append(w)
        line_heights.append(h)

    padding_x = 24
    padding_y = 12
    total_w = min(video_w - 40, max(line_widths) + padding_x*2)
    total_h = sum(line_heights) + padding_y*(len(lines)+1)

    # create RGBA image for subtitle block
    img = Image.new("RGBA", (int(total_w), int(total_h)), (0,0,0,0))
    d = ImageDraw.Draw(img)

    # draw semi-transparent rounded-ish rectangle (simple rectangle)
    d.rectangle([0, 0, img.width, img.height], fill=(0,0,0,160))

    # draw lines, right-aligned inside the rectangle
    y = padding_y
    for i, line in enumerate(lines):
        lw = line_widths[i]
        x = img.width - padding_x - lw  # right align
        try:
            d.text((x, y), line, font=font, fill=(255,255,255,255), stroke_width=2, stroke_fill=(0,0,0,255))
        except TypeError:
            # older pillow fallback (outline manually)
            outline_color = (0,0,0,255)
            for ox, oy in [(-1,0),(1,0),(0,-1),(0,1)]:
                d.text((x+ox, y+oy), line, font=font, fill=outline_color)
            d.text((x, y), line, font=font, fill=(255,255,255,255))
        y += line_heights[i] + padding_y

    return img

# ---------------------------
# Burn subtitles onto video (single image overlay that contains multiple lines)
# ---------------------------
def burn_subtitles_on_video(video_path: str, translated_text: str):
    clip = VideoFileClip(video_path)
    w, h = clip.w, clip.h

    subtitle_img = create_subtitle_image(translated_text, w, h, fontsize=max(20, int(w/36)))
    subtitle_np = np.array(subtitle_img)

    # place at bottom center with some margin
    subtitle_clip = ImageClip(subtitle_np)\
        .set_duration(clip.duration)\
        .set_position(("center", h - subtitle_img.height - 30))

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
    except Exception:
        pass

# ---------------------------
#  Telegram handler
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

        # duration check
        clip = VideoFileClip(tmp_video.name)
        if clip.duration > 5 * 60 + 5:
            bot.reply_to(message, "❌ הסרטון ארוך מדי — המקסימום הוא 5 דקות.")
            clip.close()
            os.remove(tmp_video.name)
            return
        clip.close()

        send_progress(chat_id, "🎧 ממיר את האודיו לטקסט...")
        text = transcribe_audio(tmp_video.name)

        if not text or not text.strip():
            bot.reply_to(message, "❌ לא זוהה דיבור בסרטון.")
            os.remove(tmp_video.name)
            return

        send_progress(chat_id, "🔠 מתרגם לעברית...")
        try:
            translated = translator.translate(text)
        except Exception:
            translated = text

        send_progress(chat_id, "🔥 שורף כתוביות על הווידאו (זה עשוי לקחת כמה דקות)...")
        out_video = burn_subtitles_on_video(tmp_video.name, translated)

        send_progress(chat_id, "📤 מעלה את הסרטון בחזרה...")
        with open(out_video, "rb") as f:
            bot.send_video(chat_id, f, caption="✅ הנה הסרטון שלך עם כתוביות בעברית.")

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
        bot.reply_to(message, f"❌ קרתה שגיאה: {e}\n{tb}")

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
