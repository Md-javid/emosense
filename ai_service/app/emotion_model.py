import os
import re
import cv2
import cv2.data
import json
import time
import base64
import numpy as np
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from PIL import Image
from transformers import pipeline

# Load environment variables from .env if present
load_dotenv()

EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
PRIMARY_MODEL = "trpakov/vit-face-expression"

_hf_pipeline = None

# Initialize robust Haar Cascades for face tracking
_face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
_face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

def get_hf_pipeline():
    """
    Lazily initializes and returns the Hugging Face emotion classification pipeline.
    """
    global _hf_pipeline
    if _hf_pipeline is None:
        try:
            print(f"[Hugging Face] Loading Vision Transformer model ({PRIMARY_MODEL})...")
            _hf_pipeline = pipeline("image-classification", model=PRIMARY_MODEL)
            print("[Hugging Face] Model loaded successfully.")
        except Exception as e:
            print(f"[Hugging Face] Error loading model {PRIMARY_MODEL}: {e}")
    return _hf_pipeline


def detect_face_bbox(gray: np.ndarray, img_w: int, img_h: int) -> tuple:
    """
    Multi-cascade face detector for robust human face tracking under diverse angles.
    """
    faces = _face_cascade_alt.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=3, minSize=(40, 40))
    if len(faces) == 0:
        faces = _face_cascade_default.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(40, 40))

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        top_face = faces[0]
        return int(top_face[0]), int(top_face[1]), int(top_face[2]), int(top_face[3])
    
    return int(img_w * 0.15), int(img_h * 0.15), int(img_w * 0.7), int(img_h * 0.7)


def analyze_opencv_facial_affect(bgr_image: np.ndarray) -> Dict[str, Any]:
    """
    Calibrated Facial Geometry & Dynamic Affect Fallback Engine.
    """
    if bgr_image is None or bgr_image.size == 0:
        return {
            "emotion": "no_face",
            "confidence": 0.0,
            "all_probs": {l: 0.0 for l in EMOTION_LABELS},
            "bbox": [0, 0, 0, 0]
        }

    img_h, img_w = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
    fx, fy, fw, fh = detect_face_bbox(gray, img_w, img_h)
    
    face_raw = gray[max(0, fy):min(img_h, fy+fh), max(0, fx):min(img_w, fx+fw)]
    if face_raw.size == 0:
        face_raw = gray

    face_norm = cv2.resize(face_raw, (120, 120), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face_eq = clahe.apply(face_norm)

    # 1. Glabella & Eyebrow vertical gradient (AU4 - Brow Lowerer)
    glabella = face_eq[20:45, 48:72]
    sobel_v = cv2.Sobel(glabella, cv2.CV_64F, 1, 0, ksize=3)
    glabella_grad = float(np.mean(np.abs(sobel_v))) if glabella.size > 0 else 0.0

    # 2. Smile & Lip Corners (AU12 Lip Corner Puller)
    mouth_roi = face_eq[72:115, 20:100]
    left_corner_zone = mouth_roi[5:35, 5:28]
    right_corner_zone = mouth_roi[5:35, 52:75]
    center_lip_zone = mouth_roi[5:35, 28:52]

    left_y_min = np.argmin(np.mean(left_corner_zone, axis=1)) if left_corner_zone.size > 0 else 15
    right_y_min = np.argmin(np.mean(right_corner_zone, axis=1)) if right_corner_zone.size > 0 else 15
    center_y_min = np.argmin(np.mean(center_lip_zone, axis=1)) if center_lip_zone.size > 0 else 15
    corner_y_avg = (left_y_min + right_y_min) / 2.0
    smile_lift = float(center_y_min - corner_y_avg)

    # Mouth opening aspect ratio
    mouth_bin = cv2.threshold(mouth_roi, 70, 255, cv2.THRESH_BINARY_INV)[1]
    mouth_v_proj = np.sum(mouth_bin > 0, axis=1)
    mouth_open_height = np.sum(mouth_v_proj > (mouth_roi.shape[1] * 0.15))
    mouth_h_proj = np.sum(mouth_bin > 0, axis=0)
    mouth_open_width = np.sum(mouth_h_proj > (mouth_roi.shape[0] * 0.15))
    mouth_aspect_ratio = float(mouth_open_height / max(1, mouth_open_width))

    # Cascade smile verification
    lower_face = face_raw[int(fh*0.45):, :] if fh > 0 else face_raw
    smiles = _smile_cascade.detectMultiScale(lower_face, scaleFactor=1.15, minNeighbors=3, minSize=(18, 18))
    smile_detected = len(smiles) > 0

    scores: Dict[str, float] = {
        "happy": 0.10,
        "sad": 0.10,
        "angry": 0.10,
        "surprise": 0.10,
        "fear": 0.08,
        "disgust": 0.08,
        "neutral": 0.35
    }

    if smile_detected or smile_lift > 1.0 or (mouth_open_width > 40 and smile_lift >= 0):
        intensity = 1.0 + max(0.0, smile_lift * 0.5) + (1.2 if smile_detected else 0.0)
        scores["happy"] += 4.0 * intensity
        scores["neutral"] = 0.05
    elif mouth_aspect_ratio > 0.40 or mouth_open_height > 16:
        intensity = (mouth_aspect_ratio * 2.5)
        scores["surprise"] += 3.5 * intensity
        scores["fear"] += 0.8 * intensity
        scores["neutral"] = 0.05
    elif glabella_grad > 14.0:
        intensity = (glabella_grad / 9.0)
        scores["angry"] += 3.2 * intensity
        scores["disgust"] += 0.6 * intensity
        scores["neutral"] = 0.05
    elif smile_lift < -1.0 and mouth_aspect_ratio < 0.35:
        intensity = max(0.0, -smile_lift * 0.5)
        scores["sad"] += 3.0 * intensity
        scores["neutral"] = 0.05

    max_label = max(scores, key=lambda k: scores[k])
    exp_scores = {k: np.exp(v * 1.5) for k, v in scores.items()}
    total_exp = sum(exp_scores.values())
    all_probs = {k: round(float(v / total_exp), 4) for k, v in exp_scores.items()}
    sorted_probs = {k: v for k, v in sorted(all_probs.items(), key=lambda x: x[1], reverse=True)}
    confidence = sorted_probs[max_label]

    return {
        "emotion": max_label,
        "confidence": confidence,
        "all_probs": sorted_probs,
        "bbox": [fx, fy, fw, fh]
    }


def normalize_emotion_response(
    raw_data: Dict[str, Any],
    image_shape: tuple,
    default_bbox: Optional[List[int]] = None
) -> Dict[str, Any]:
    """
    Normalizes the parsed JSON output to guarantee strict
    adherence to the frontend EmotionPrediction interface.
    """
    img_h, img_w = image_shape[:2]
    
    raw_emotion = str(raw_data.get("emotion", "neutral")).strip().lower()
    emotion = raw_emotion if raw_emotion in EMOTION_LABELS else "neutral"
    
    try:
        raw_conf = raw_data.get("confidence", 0.0)
        confidence = float(raw_conf) if raw_conf is not None else 0.85
        confidence = max(0.40, min(0.99, confidence))
    except (ValueError, TypeError):
        confidence = 0.85

    raw_probs = raw_data.get("all_probs", {})
    all_probs: Dict[str, float] = {}
    
    if isinstance(raw_probs, dict) and len(raw_probs) > 0:
        for label in EMOTION_LABELS:
            val = raw_probs.get(label, raw_probs.get(label.capitalize(), 0.0))
            if val is not None:
                try:
                    all_probs[label] = max(0.0, min(1.0, float(val)))
                except (ValueError, TypeError):
                    all_probs[label] = 0.0
            else:
                all_probs[label] = 0.0
    
    total_p = sum(all_probs.values())
    if total_p <= 0.0:
        remainder = round((1.0 - confidence) / max(1, (len(EMOTION_LABELS) - 1)), 4)
        for label in EMOTION_LABELS:
            all_probs[label] = confidence if label == emotion else remainder
    else:
        all_probs = {k: round(v / total_p, 4) for k, v in all_probs.items()}
    
    sorted_probs = {k: v for k, v in sorted(all_probs.items(), key=lambda item: item[1], reverse=True)}

    raw_bbox = raw_data.get("bbox", None) or default_bbox
    bbox: List[int] = [int(img_w * 0.15), int(img_h * 0.15), int(img_w * 0.7), int(img_h * 0.7)]
    if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4:
        try:
            bx, by, bw, bh = [int(v) for v in raw_bbox]
            bx = max(0, min(img_w - 1, bx))
            by = max(0, min(img_h - 1, by))
            bw = max(1, min(img_w - bx, bw))
            bh = max(1, min(img_h - by, bh))
            bbox = [bx, by, bw, bh]
        except (ValueError, TypeError):
            pass

    return {
        "emotion": emotion,
        "confidence": confidence,
        "all_probs": sorted_probs,
        "bbox": bbox
    }


def predict_emotion(bgr_image: np.ndarray, is_static_upload: bool = False) -> Dict[str, Any]:
    """
    Direct Vision Transformer Affective Inference:
    Executes Hugging Face ViT model on every frame / image, with facial crop enhancement.
    """
    if bgr_image is None or bgr_image.size == 0:
        return {
            "emotion": "no_face",
            "confidence": 0.0,
            "all_probs": {l: 0.0 for l in EMOTION_LABELS},
            "bbox": [0, 0, 0, 0]
        }

    orig_h, orig_w = bgr_image.shape[:2]
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    detected_bbox = detect_face_bbox(gray, orig_w, orig_h)
    
    hf_pipe = get_hf_pipeline()
    if hf_pipe is not None:
        try:
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            fx, fy, fw, fh = detected_bbox
            
            # Crop with padding for optimal Vision Transformer classification
            pad_w = int(fw * 0.12)
            pad_h = int(fh * 0.12)
            x1 = max(0, fx - pad_w)
            y1 = max(0, fy - pad_h)
            x2 = min(orig_w, fx + fw + pad_w)
            y2 = min(orig_h, fy + fh + pad_h)
            
            face_crop = rgb_image[y1:y2, x1:x2]
            if face_crop.size == 0 or face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
                face_crop = rgb_image
                
            pil_image = Image.fromarray(face_crop)
            predictions = hf_pipe(pil_image)
            
            parsed_json = {"all_probs": {}}
            for pred in predictions:
                label = pred['label'].lower()
                parsed_json["all_probs"][label] = float(pred['score'])
            
            if predictions:
                parsed_json["emotion"] = predictions[0]['label'].lower()
                parsed_json["confidence"] = float(predictions[0]['score'])

            return normalize_emotion_response(parsed_json, (orig_h, orig_w), default_bbox=list(detected_bbox))
        except Exception as err:
            print(f"[Hugging Face Vision] Inference error ({err}). Using facial geometry fallback.")

    return analyze_opencv_facial_affect(bgr_image)


def predict_emotion_from_path(image_path: str, is_static_upload: bool = True) -> Dict[str, Any]:
    """
    Reads an image from disk and runs emotion prediction.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    bgr_image = cv2.imread(image_path)
    if bgr_image is None:
        raise ValueError(f"Could not read image at {image_path}")
    return predict_emotion(bgr_image, is_static_upload=is_static_upload)
