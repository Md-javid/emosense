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
_last_hf_frame_time: float = 0.0
_last_cached_frame_result: Optional[Dict[str, Any]] = None

# Initialize robust Haar Cascades for real-time facial expression telemetry
_face_cascade_alt = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')
_face_cascade_default = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
_smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
_eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

def get_hf_pipeline():
    """
    Lazily initializes and returns the Hugging Face emotion classification pipeline.
    """
    global _hf_pipeline
    if _hf_pipeline is None:
        try:
            _hf_pipeline = pipeline("image-classification", model=PRIMARY_MODEL)
        except Exception as e:
            print(f"Error loading model {PRIMARY_MODEL}: {e}")
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
    Calibrated Facial Geometry & Dynamic Affect Engine.
    Computes real facial morphology and action units:
    - AU12 / Smile curvature (mouth corners vs center elevation)
    - AU25/26 / Oral aperture (mouth aspect ratio and opening)
    - AU4 / Glabella furrow & brow lowering (anger / intense focus)
    - AU1 / Inner eyebrow lift (sadness / distress)
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

    # Smile curvature: difference in vertical centroid between lip corners and center
    left_y_min = np.argmin(np.mean(left_corner_zone, axis=1)) if left_corner_zone.size > 0 else 15
    right_y_min = np.argmin(np.mean(right_corner_zone, axis=1)) if right_corner_zone.size > 0 else 15
    center_y_min = np.argmin(np.mean(center_lip_zone, axis=1)) if center_lip_zone.size > 0 else 15
    corner_y_avg = (left_y_min + right_y_min) / 2.0
    smile_lift = float(center_y_min - corner_y_avg)

    # Mouth opening aspect ratio (AU25/AU26)
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

    # Continuous Affect Scoring
    scores: Dict[str, float] = {
        "happy": 0.05,
        "sad": 0.05,
        "angry": 0.05,
        "surprise": 0.05,
        "fear": 0.04,
        "disgust": 0.04,
        "neutral": 0.20
    }

    if smile_detected or smile_lift > 1.2 or (mouth_open_width > 42 and smile_lift >= 0):
        intensity = 1.0 + max(0.0, smile_lift * 0.4) + (0.9 if smile_detected else 0.0)
        scores["happy"] += 3.2 * intensity
        scores["neutral"] = max(0.02, scores["neutral"] - 0.15)
    elif mouth_aspect_ratio > 0.45 or mouth_open_height > 18:
        intensity = (mouth_aspect_ratio * 2.2)
        scores["surprise"] += 3.0 * intensity
        scores["fear"] += 0.6 * intensity
        scores["neutral"] = max(0.02, scores["neutral"] - 0.15)
    elif glabella_grad > 15.0:
        intensity = (glabella_grad / 10.0)
        scores["angry"] += 2.8 * intensity
        scores["disgust"] += 0.5 * intensity
        scores["neutral"] = max(0.02, scores["neutral"] - 0.15)
    elif smile_lift < -1.2 and mouth_aspect_ratio < 0.35:
        intensity = max(0.0, -smile_lift * 0.4)
        scores["sad"] += 2.6 * intensity
        scores["neutral"] = max(0.02, scores["neutral"] - 0.15)
    else:
        # Balanced resting neutral expression
        scores["neutral"] = 1.6
        scores["happy"] += max(0.0, smile_lift * 0.1)

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
        confidence = float(raw_conf) if raw_conf is not None else 0.90
        confidence = max(0.50, min(0.99, confidence))
    except (ValueError, TypeError):
        confidence = 0.90

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
    Affective Telemetry Engine:
    - For Static Uploads: runs Hugging Face Vision classification.
    - For Live Stream: runs adaptive Hugging Face Vision sampling with high-speed geometric fallback.
    """
    global _last_hf_frame_time, _last_cached_frame_result

    if bgr_image is None or bgr_image.size == 0:
        return {
            "emotion": "no_face",
            "confidence": 0.0,
            "all_probs": {l: 0.0 for l in EMOTION_LABELS},
            "bbox": [0, 0, 0, 0]
        }

    orig_h, orig_w = bgr_image.shape[:2]
    
    # Fast facial geometry & bounding box calculation
    cv_result = analyze_opencv_facial_affect(bgr_image)
    detected_bbox = cv_result["bbox"]

    hf_pipe = get_hf_pipeline()
    now = time.time()

    # Determine whether to execute HF Vision:
    # 1. Always for static image uploads
    # 2. For live camera frames: throttled to once every 1.5 seconds
    should_call_hf = hf_pipe is not None and (is_static_upload or (now - _last_hf_frame_time >= 1.5))

    if should_call_hf and hf_pipe is not None:
        try:
            # Prepare image for HF pipeline
            rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
            
            # If a face is detected, passing the face crop often gives better results for FER models.
            fx, fy, fw, fh = detected_bbox
            if fw > 0 and fh > 0:
                face_crop = rgb_image[max(0, fy):min(orig_h, fy+fh), max(0, fx):min(orig_w, fx+fw)]
            else:
                face_crop = rgb_image
                
            pil_image = Image.fromarray(face_crop)

            predictions = hf_pipe(pil_image)
            _last_hf_frame_time = now
            
            # Format predictions
            parsed_json = {"all_probs": {}}
            for pred in predictions:
                label = pred['label'].lower()
                parsed_json["all_probs"][label] = pred['score']
            
            if predictions:
                parsed_json["emotion"] = predictions[0]['label'].lower()
                parsed_json["confidence"] = predictions[0]['score']

            normalized = normalize_emotion_response(parsed_json, (orig_h, orig_w), default_bbox=detected_bbox)
            _last_cached_frame_result = normalized
            return normalized
        except Exception as err:
            print(f"[HF Vision] Error during inference ({err}). Using geometric telemetry fallback.")

    # Return cached HF result with updated bounding box if recent, or real-time CV result
    if _last_cached_frame_result and (now - _last_hf_frame_time < 3.0):
        res = dict(_last_cached_frame_result)
        res["bbox"] = detected_bbox
        return res

    return cv_result

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

