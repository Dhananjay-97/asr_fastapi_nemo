# Hindi ASR API with FastAPI and NVIDIA NeMo

This project provides a FastAPI REST API for Hindi speech-to-text (ASR) using NVIDIA NeMo's Conformer-CTC model (`stt_hi_conformer_ctc_medium`). The model is downloaded via NeMo, exported to ONNX for optimized inference, and served using FastAPI.

## Project Structure

## Prerequisites

*   Python 3.9+
*   Docker (for containerized deployment)
*   Access to NVIDIA NGC for NeMo model download (usually handled automatically by NeMo if you have an NGC API key configured, or models are public)

## Setup & Build

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd asr_fastapi_nemo
    ```

2.  **(Optional) Create a Python virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Export the NeMo model to ONNX:**
    This is a one-time step. It downloads the specified NeMo ASR model and converts it to the ONNX format, saving it in the `model/` directory.
    ```bash
    python model/export_model.py
    ```
    This will create `model/stt_hi_conformer_ctc_medium.onnx`.

4.  **Build the Docker Container:**
    ```bash
    docker build -t hi-asr-api .
    ```
    Alternatively, use the Makefile:
    ```bash
    make build
    ```

## Running the Application

### Using Docker (Recommended)

```bash
docker run -p 8000:8000 -v $(pwd)/model:/home/appuser/app/model hi-asr-api

make run

uvicorn app.main:app --reload --port 8000

curl -X POST http://localhost:8000/transcribe \
     -F "file=@audio/test_hi.wav"

{
  "transcription": "नमस्ते दुनिया"
}

curl http://localhost:8000/health

{"status":"ok","model_path":"model/stt_hi_conformer_ctc_medium.onnx"}
