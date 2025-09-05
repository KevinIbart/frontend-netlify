from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from deepface import DeepFace
import cv2
import numpy as np

app = FastAPI()

# Configuración CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Restringir en producción
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#Hola
@app.post("/analizar/")
async def analizar(file: UploadFile = File(...)):
    try:
        # Leer imagen
        image_bytes = await file.read()
        np_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

        # Analizar emociones
        result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

        # DeepFace puede devolver dict o lista
        if isinstance(result, list):
            data = result[0]
        else:
            data = result

        # Si no hay emociones detectadas
        if "emotion" not in data or "dominant_emotion" not in data:
            return {
                "status": "error",
                "mensaje": "No se detectó un rostro",
                "emocion_dominante": None,
                "emociones": {}
            }

        # Convertir numpy.float32 → float
        emociones = {emo: float(valor) for emo, valor in data["emotion"].items()}

        return {
            "status": "success",
            "mensaje": "Rostro detectado y analizado",
            "emocion_dominante": str(data["dominant_emotion"]),
            "emociones": emociones
        }

    except Exception as e:
        return {"status": "error", "mensaje": f"Error en análisis: {str(e)}"}
