FROM python:3.11-slim

WORKDIR /app

# install ffmpeg, fontconfig, and libass-dev (for robust SRT/RTL subtitle burning)
RUN apt-get update && \
    apt-get install -y ffmpeg fontconfig libass-dev && \
    rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# copy fonts folder to /app/fonts/ (using your original, correct path)
COPY fonts/ /app/fonts/

# **חשוב:** רענון מטמון הגופנים כדי ש-libass/ffmpeg יזהו את הגופן המותאם אישית
RUN fc-cache -f -v

EXPOSE 8080

CMD ["python", "app.py"]
