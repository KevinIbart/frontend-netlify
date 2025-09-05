# Imagen base oficial de Python
FROM python:3.10-slim

# Evitar buffering de logs en Render
ENV PYTHONUNBUFFERED=1

# Crear directorio de la app
WORKDIR /app

# Instalar dependencias del sistema necesarias para deepface, opencv y numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de dependencias primero para cache
COPY requirements.txt .

# Actualizar pip e instalar dependencias
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la app
COPY . .

# Exponer puerto que Render usará
EXPOSE 8000

# Comando para ejecutar con Gunicorn + Uvicorn (FastAPI)
CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "-b", "0.0.0.0:8000", "main:app"]
