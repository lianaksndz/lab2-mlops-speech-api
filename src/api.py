import io
import json
import os
import time
from pathlib import Path

import soundfile as sf
import torch

from fastapi import FastAPI, File, HTTPException, UploadFile

from model import (
    SmallCNN,
    CLASSES,
    create_mel_transform,
    prepare_audio
)

MODEL_PATH = Path(os.getenv("MODEL_PATH", "artifacts/model.pt"))
METRICS_PATH = Path(os.getenv("METRICS_PATH", "artifacts/metrics.json"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")

app = FastAPI(
    title="Speech Commands ML API",
    description="API для розпізнавання голосових команд",
    version=MODEL_VERSION
)

device = torch.device("cpu")
mel_transform = create_mel_transform()
model = None
model_metrics = {}


@app.on_event("startup")
def load_model():
    global model, model_metrics

    if not MODEL_PATH.exists():
        raise RuntimeError(f"Файл моделі не знайдено: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    classes = checkpoint.get("classes", CLASSES)

    if classes != CLASSES:
        raise RuntimeError(f"Несумісні класи моделі: {classes}")

    model = SmallCNN(num_classes=len(CLASSES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as file:
            model_metrics = json.load(file)


@app.get("/")
def root():
    return {
        "message": "Speech Commands ML API is running",
        "version": MODEL_VERSION
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "unavailable",
        "model_loaded": model is not None
    }


@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель ще не завантажена"
        )

    model_size = MODEL_PATH.stat().st_size

    return {
        "model_name": "SmallCNN",
        "model_version": MODEL_VERSION,
        "classes": CLASSES,
        "sample_rate": 16000,
        "model_size_kb": round(model_size / 1024, 2),
        "accuracy_percent": model_metrics.get("accuracy_percent")
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Модель недоступна"
        )

    if not file.filename or not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Потрібно завантажити WAV-файл"
        )

    contents = await file.read()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="Файл порожній"
        )

    try:
        start_time = time.perf_counter()

        # Читання WAV через soundfile без тимчасового файлу.
        audio, sample_rate = sf.read(
            io.BytesIO(contents),
            dtype="float32",
            always_2d=True
        )

        # Перетворення форми (семпли, канали)
        # у форму (канали, семпли), яку очікує модель.
        waveform = torch.from_numpy(audio.T.copy())

        inputs = prepare_audio(
            waveform,
            sample_rate,
            mel_transform
        ).to(device)

        with torch.no_grad():
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_index = torch.max(
            probabilities,
            dim=1
        )

        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        return {
            "prediction": CLASSES[predicted_index.item()],
            "confidence": round(confidence.item(), 4),
            "probabilities": {
                name: round(
                    probabilities[0][index].item(),
                    4
                )
                for index, name in enumerate(CLASSES)
            },
            "latency_ms": round(latency_ms, 3),
            "model_version": MODEL_VERSION
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Не вдалося обробити WAV-файл: {error}"
        )