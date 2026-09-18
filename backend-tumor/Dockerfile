# Gunakan Python 3.10 slim (stabil untuk AI)
FROM python:3.10-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Instal dependensi sistem
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Salin file requirements
COPY requirements.txt .

# Instal dependensi Python
RUN pip install --no-cache-dir \
    --timeout=120 \
    --retries=5 \
    -r requirements.txt

# Salin sisa kode backend
COPY . .

# Ekspos port 8000 agar bisa diakses dari luar
EXPOSE 8000

# Jalankan FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]