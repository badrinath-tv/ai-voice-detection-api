from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import base64
import librosa
import numpy as np
import io
import os
import joblib

app = FastAPI(title="AI Voice Detection API")

# ==================================
# Enable CORS
# ==================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================================
# Root Health Check
# ==================================
@app.get("/")
def root():
    return {
        "status": "running",
        "message": "AI Voice Detection API is operational"
    }

# ==================================
# Load Trained Model
# ==================================
MODEL_PATH = "voice_model.pkl"

if not os.path.exists(MODEL_PATH):
    raise RuntimeError("voice_model.pkl not found")

model = joblib.load(MODEL_PATH)

# ==================================
# API Key
# ==================================
API_KEY = os.getenv("API_KEY", "12345")

# ==================================
# Feature Extraction
# ==================================
def extract_features(audio_bytes):

    try:
        audio, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=16000,
            duration=3
        )

        # Ensure minimum length
        if len(audio) < 1000:
            return None

        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)

        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_var = np.var(mfcc, axis=1)

        features = np.concatenate((mfcc_mean, mfcc_var))

        return features.reshape(1, -1)

    except Exception:
        return None

# ==================================
# Voice Detection Endpoint
# ==================================
@app.post("/api/voice-detection")
def voice_detection(payload: dict, x_api_key: str = Header(None)):

    # 🔐 API Key Validation
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 📦 Required Fields Validation
    required_fields = ["language", "audioFormat", "audioBase64"]

    for field in required_fields:
        if field not in payload:
            raise HTTPException(status_code=400, detail=f"Missing field: {field}")

    if payload["audioFormat"].lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only MP3 format supported")

    # 🔓 Decode Base64 Audio
    try:
        audio_bytes = base64.b64decode(payload["audioBase64"])
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 audio data")

    # 🎛 Feature Extraction
    features = extract_features(audio_bytes)

    if features is None:
        raise HTTPException(status_code=422, detail="Audio too short or corrupted")

    # 🤖 Model Prediction
    try:
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        ai_prob = float(probabilities[1])
        human_prob = float(probabilities[0])

        if prediction == 1:
            classification = "AI_GENERATED"
            confidence = ai_prob
            explanation = "Detected synthetic acoustic patterns with uniform spectral structure"
        else:
            classification = "HUMAN"
            confidence = human_prob
            explanation = "Detected natural human speech dynamics and acoustic variability"

    except Exception:
        raise HTTPException(status_code=500, detail="Model inference failed")

    # 📤 Response
    return {
        "status": "success",
        "language": payload["language"],
        "classification": classification,
        "confidenceScore": round(confidence, 2),
        "explanation": explanation
    }
