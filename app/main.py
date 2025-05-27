# File: app/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.concurrency import run_in_threadpool
import soundfile as sf
import io
import numpy as np
import os
from contextlib import asynccontextmanager

from app.model import SpeechRecognizer

# Configuration
MODEL_ONNX_PATH = os.getenv("MODEL_ONNX_PATH", "model/stt_hi_conformer_ctc_medium.onnx")
NEMO_MODEL_NAME = os.getenv("NEMO_MODEL_NAME", "stt_hi_conformer_ctc_medium")
MAX_AUDIO_DURATION_SECONDS = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "10"))
EXPECTED_SAMPLE_RATE = 16000

# Global recognizer instance, initialized at startup
recognizer_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ASR model during startup
    global recognizer_instance
    if not os.path.exists(MODEL_ONNX_PATH):
        raise FileNotFoundError(
            f"ONNX model not found at {MODEL_ONNX_PATH}. "
            "Please run model/export_model.py first."
        )
    recognizer_instance = SpeechRecognizer(model_path=MODEL_ONNX_PATH, nemo_model_name=NEMO_MODEL_NAME)
    yield
    # Clean up resources if needed on shutdown (not strictly necessary here)
    recognizer_instance = None


app = FastAPI(lifespan=lifespan)

def get_recognizer():
    if recognizer_instance is None:
        # This should not happen if lifespan is used correctly
        raise HTTPException(status_code=503, detail="ASR model not initialized")
    return recognizer_instance

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    recognizer: SpeechRecognizer = Depends(get_recognizer)
):
    # 1. Validate file type
    if file.content_type != "audio/wav" and not file.filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only WAV audio files are allowed."
        )

    # 2. Read audio content asynchronously
    try:
        audio_bytes = await file.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading uploaded file: {e}")
    finally:
        await file.close()

    # 3. Decode WAV bytes and validate audio properties
    try:
        # Use soundfile to read audio data from bytes
        audio_np, sr = sf.read(io.BytesIO(audio_bytes))
    except Exception as e: # Could be sf.LibsndfileError or RuntimeError
        raise HTTPException(status_code=400, detail=f"Invalid WAV file. Could not decode. Error: {e}")

    if sr != EXPECTED_SAMPLE_RATE:
        # For simplicity, we reject. In a real app, you might resample.
        raise HTTPException(
            status_code=400,
            detail=f"Audio sample rate must be {EXPECTED_SAMPLE_RATE} Hz. Received {sr} Hz."
        )

    # Ensure audio is 1D (mono)
    if audio_np.ndim > 1 and audio_np.shape[1] == 2: # Stereo
        audio_np = audio_np.mean(axis=1) # Convert to mono by averaging channels
    elif audio_np.ndim > 1:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported number of audio channels: {audio_np.shape[1]}. Only mono or stereo (auto-converted to mono) supported."
        )
        
    duration_seconds = len(audio_np) / sr
    if duration_seconds == 0:
        raise HTTPException(status_code=400, detail="Audio file is empty or too short.")
    if duration_seconds > MAX_AUDIO_DURATION_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Audio duration exceeds maximum of {MAX_AUDIO_DURATION_SECONDS} seconds. Received {duration_seconds:.2f}s."
        )

    # 4. Run inference in a thread pool (since ONNX inference and NeMo preprocessing are CPU-bound)
    try:
        transcription = await run_in_threadpool(recognizer.predict, audio_np, sr)
    except Exception as e:
        # Log the exception for server-side debugging
        app.logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error during transcription: {e}")

    return {"transcription": transcription}

@app.get("/health")
async def health_check():
    if recognizer_instance is None:
        return {"status": "initializing_asr_model"}
    return {"status": "ok", "model_path": MODEL_ONNX_PATH}

# To run locally (for development):
# uvicorn app.main:app --reload --port 8000