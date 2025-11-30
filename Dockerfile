FROM python:3.11-slim

# הגדרת סביבת עבודה
WORKDIR /app

# 1. התקנת Ffmpeg, Fontconfig, וספריית libass לתמיכה בכתוביות
# libass-dev: נחוץ לצריבת כתוביות SRT עם גופנים מותאמים אישית (RTL)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        fontconfig \
        libass-dev && \
    rm -rf /var/lib/apt/lists/*

# 2. העתקת דרישות והתקנת ספריות Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. העתקת גופנים למיקום נגיש
# נתיב זה (/usr/share/fonts/truetype/custom) נגיש יותר ל-libass
COPY fonts/NotoSansHebrew.ttf /usr/share/fonts/truetype/custom/
RUN fc-cache -f -v # רענן את מטמון הגופנים

# 4. העתקת הקוד והפעלת היישום
COPY . .

EXPOSE 8080

CMD ["python", "app.py"]
