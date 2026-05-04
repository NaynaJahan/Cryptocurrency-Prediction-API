# Use slim Python image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (optional but helpful for TLS/CA and timezones)
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates tzdata && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# If you want to install your TestPyPI package in the container:
# RUN pip install --no-cache-dir -i https://test.pypi.org/simple/ amla-at1==2025.0.3.0

# Copy code + model
COPY app ./app
COPY models ./models

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]