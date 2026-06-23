"""
DeepFace Analyzer — Versión para estudiantes
"""

from deepface import DeepFace
import cv2
import numpy as np
import base64
from pathlib import Path
import os


# ================== ETIQUETAS EN ESPAÑOL ==================
GENDER_LABELS_ES = {
    "Man":   "Hombre",
    "Woman": "Mujer",
}

RACE_LABELS_ES = {
    "asian":           "Asiático",
    "indian":          "Indio",
    "black":           "Afrodescendiente",
    "white":           "Caucásico",
    "middle eastern":  "Oriente Medio",
    "latino hispanic": "Latino / Hispano",
}

# Tarea 2 — emociones traducidas al español
EMOTION_LABELS_ES = {
    "angry":    "Enojado",
    "disgust":  "Asco",
    "fear":     "Miedo",
    "happy":    "Feliz",
    "sad":      "Triste",
    "surprise": "Sorprendido",
    "neutral":  "Neutral",
}


def convert_numpy_to_python(obj):
    """Convierte tipos numpy a Python nativo (necesario para JSON)."""
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {convert_numpy_to_python(k): convert_numpy_to_python(v)
                for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_to_python(item) for item in obj]
    else:
        return obj


def analyze_image(image_path: str | Path) -> list[dict]:
    """Analiza una imagen y retorna lista de rostros detectados."""
    try:
        results = DeepFace.analyze(
            img_path=str(image_path),
            actions=["age", "gender", "emotion", "race"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        if not results:
            raise ValueError("No se detectó ningún rostro.")
    except Exception as e:
        raise ValueError(f"Error al analizar la imagen: {str(e)}")

    faces = []
    for r in results:
        r_clean = convert_numpy_to_python(r)
        dominant_gender  = r_clean["dominant_gender"]
        dominant_race    = r_clean["dominant_race"]
        dominant_emotion = r_clean.get("dominant_emotion", "neutral")

        face_data = {
            "genero":            GENDER_LABELS_ES.get(dominant_gender, dominant_gender),
            "genero_confianza":  round(float(r_clean["gender"][dominant_gender]), 1),
            "raza_dominante":    RACE_LABELS_ES.get(dominant_race, dominant_race),
            # Datos adicionales — los estudiantes pueden mostrarlos
            "edad_estimada":     int(r_clean.get("age", 0)),
            "emocion":           EMOTION_LABELS_ES.get(dominant_emotion, dominant_emotion),
            "emociones_detalle": [
                {
                    "nombre":     EMOTION_LABELS_ES.get(k, k),
                    "porcentaje": round(float(v), 1)
                }
                for k, v in r_clean.get("emotion", {}).items()
            ],
            "razas_detalle": {
                k: round(float(v), 1)
                for k, v in r_clean.get("race", {}).items()
            },
            "region": r_clean.get("region", {})
        }
        faces.append(face_data)
    return faces


def analyze_from_bytes(image_bytes: bytes) -> tuple[list[dict], str]:
    """Analiza desde bytes — usado por la app web Flask."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen.")

    tmp_path = "/tmp/deepface_temp.jpg"
    cv2.imwrite(tmp_path, img)
    try:
        faces = analyze_image(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    annotated = draw_annotations(img.copy(), faces)
    _, buffer  = cv2.imencode(".jpg", annotated)
    img_b64    = base64.b64encode(buffer).decode("utf-8")
    return faces, img_b64


def draw_annotations(img: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Dibuja rectángulos alrededor de los rostros detectados."""
    for i, face in enumerate(faces, start=1):
        r = face.get("region", {})
        if not r:
            continue
        x, y, w, h = int(r.get("x", 0)), int(r.get("y", 0)), int(r.get("w", 0)), int(r.get("h", 0))
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 200, 100), 3)
        cv2.putText(img, f"Rostro {i}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return img
