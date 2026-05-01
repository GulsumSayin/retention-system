FROM python:3.11-slim

WORKDIR /app

# Sistem bağımlılıkları (bazı ML kütüphaneleri için gerekli)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Önce bağımlılıkları kur — Docker layer cache'i etkin kullan
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama kodunu kopyala
COPY . .

EXPOSE 5000

# Üretim sunucusu: 2 worker, 120s zaman aşımı (ML pipeline yavaş olabilir)
CMD ["gunicorn", "flask_app:app", \
     "--workers", "2", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:5000", \
     "--access-logfile", "-", \
     "--error-logfile", "-"]
