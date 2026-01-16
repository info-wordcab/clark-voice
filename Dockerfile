# Clark Receptionist - Pipecat Voice Bot
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (ffmpeg for audio processing)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy application code
COPY . .

EXPOSE 7860

CMD ["python", "bot.py", "-t", "twilio", "--host", "0.0.0.0"]
