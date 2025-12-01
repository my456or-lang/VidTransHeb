טּFROM python:3.11-slim

WORKDIR /app

# install ffmpeg (system), fontconfig (font resolution)
RUN apt-get update && \
    apt-get install -y ffmpeg fontconfig && \
    rm -rf /var/lib/apt/lists/*

# copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy app
COPY . .

# copy fonts folder
COPY fonts/ /app/fonts/

EXPOSE 8080

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8080", "app:app"]
