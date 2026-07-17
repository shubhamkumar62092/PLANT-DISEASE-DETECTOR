import os
import json
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input

app = FastAPI(title="Plant Disease Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "vgg16_plant_disease_model.keras")
CLASS_NAMES_PATH = os.environ.get("CLASS_NAMES_PATH", "class_names.json")

model = None
class_names = []

@app.on_event("startup")
def load_model():
    global model, class_names

    if not os.path.exists(MODEL_PATH):
        print(f"[WARNING] Model file not found at {MODEL_PATH}. /predict will fail.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[INFO] Model loaded from {MODEL_PATH}")

    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH) as f:
            class_names = json.load(f)
        print(f"[INFO] {len(class_names)} classes loaded.")
    else:
        print("[WARNING] class_names.json not found.")


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    return arr


def format_class_name(raw: str) -> dict:
    parts = raw.split("___")
    plant = parts[0].replace("_", " ").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Unknown"
    return {"plant": plant, "disease": disease, "raw": raw}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "num_classes": len(class_names),
    }


@app.get("/")
def serve_frontend():
    index_path = os.path.join(os.path.dirname(__file__), "index.html")
    return FileResponse(index_path)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        raise HTTPException(status_code=400, detail="Upload a JPEG or PNG image.")

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 10 MB).")

    try:
        arr = preprocess_image(image_bytes)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    preds = model.predict(arr, verbose=0)[0]
    top_index = np.argsort(preds)[::-1][:1]   # ✅ only top 1

    results = []
    for idx in top_index:
        raw_name = class_names[idx] if idx < len(class_names) else str(idx)
        info = format_class_name(raw_name)
        results.append({
            **info,
            "confidence": round(float(preds[idx]) * 100, 2),
        })

    return {"predictions": results}


app.mount("/", StaticFiles(directory=os.path.dirname(__file__), html=True), name="static")