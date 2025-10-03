# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalación de dependencias del sistema (opcional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Actualizar pip/setuptools/wheel para mejores binarios precompilados
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copiar requerimientos e instalar
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copiar código
COPY . /app

# Crear directorio de datos para SQLite
RUN mkdir -p /data

# Variables por defecto (sobreescribir en despliegue)
ENV ALERTA_DB_PATH=/data/alertas.db \
    ALERTA_API_KEYS=alertaraven_mobile_key_2024 \
    ALERTA_STATIC_DIR=/app/static

EXPOSE 8000

# Render establece la variable PORT; usa 8000 por defecto local
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]