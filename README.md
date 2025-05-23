# asr_fastapi_nemo

asr_fastapi_nemo/
├── app/
│   ├── main.py            # FastAPI app with /transcribe endpoint
│   └── model.py           # ONNX model loader and inference class
├── model/
│   └── stt_hi_conformer_ctc_medium.onnx  # ONNX-exported ASR model
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container build instructions
├── README.md              # Setup, build, and test instructions
└── Description.md         # Features, issues, limitations, assumptions

# Hindi ASR API with FastAPI and NVIDIA NeMo

This project provides a REST API for Hindi speech recognition using NVIDIA NeMo’s Conformer-CTC model. The model is exported to ONNX for efficient inference.

## Setup & Build

1. **Model Export**: (One-time) Export the NeMo model to ONNX:
   ```bash
   python model/export_model.py
   ```
2. Build Docker Container:
   ```bash
   docker build -t hi-asr-api .
   ```
3. Run Container:
   ```bash
   docker run -p 8000:8000 hi-asr-api
   ```
## Usage:
```bash
curl -X POST http://localhost:8000/transcribe \
     -F 'file=@/path/to/hindi_sample.wav'
```
