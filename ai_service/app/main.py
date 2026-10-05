import os
import traceback
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .utils import read_upload_image, decode_base64_image
from .emotion_model import predict_emotion, get_hf_pipeline, EMOTION_LABELS

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[EmoSense AI] Preloading Hugging Face Vision Transformer model...")
    try:
        get_hf_pipeline()
        print("[EmoSense AI] Model loaded successfully and ready for inference.")
    except Exception as e:
        print(f"[EmoSense AI] Startup preload warning: {e}")
    yield

app = FastAPI(
    title="EmoSense AI Microservice",
    description="Facial Emotion Recognition microservice powered by Hugging Face Vision AI",
    version="2.0.0",
    lifespan=lifespan
)

# Enable CORS for local backend & frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FramePredictionRequest(BaseModel):
    image_base64: str

class EmotionPredictionResponse(BaseModel):
    emotion: str
    confidence: float
    all_probs: Optional[Dict[str, float]] = {}
    bbox: Optional[List[int]] = None

@app.get("/")
def root():
    return {
        "service": "EmoSense AI Microservice (Hugging Face Vision)",
        "status": "online",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "predict_frame": "/predict_frame",
            "predict_image": "/predict_image"
        },
        "model": "trpakov/vit-face-expression"
    }

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "EmoSense AI Microservice",
        "engine": "Hugging Face Vision",
        "labels": EMOTION_LABELS
    }

@app.post("/predict_image", response_model=EmotionPredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        image = read_upload_image(file_bytes)
        result = predict_emotion(image, is_static_upload=True)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Failed to process image: {str(e)}")

@app.post("/predict_frame", response_model=EmotionPredictionResponse)
def predict_frame(request: FramePredictionRequest):
    try:
        if not request.image_base64 or len(request.image_base64.strip()) < 20:
            return {
                "emotion": "no_face",
                "confidence": 0.0,
                "all_probs": {l: 0.0 for l in EMOTION_LABELS},
                "bbox": [0, 0, 0, 0]
            }
        image = decode_base64_image(request.image_base64)
        result = predict_emotion(image)
        return result
    except Exception as e:
        traceback.print_exc()
        return {
            "emotion": "no_face",
            "confidence": 0.0,
            "all_probs": {l: 0.0 for l in EMOTION_LABELS},
            "bbox": [0, 0, 0, 0]
        }
