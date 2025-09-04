FROM python:3.9-slim

# Instalar dependencias del sistema necesarias para OpenCV y DeepFace
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Crear directorio de la app
WORKDIR /app

# Copiar requirements primero para aprovechar cache
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Exponer el puerto
EXPOSE 8000

# Comando de ejecución en Render
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
