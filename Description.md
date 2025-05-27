
**`Description.md`**
(Based on OCR page 6, with formatting and reference handling)

```markdown
# File: Description.md
# Implementation Details for Hindi ASR FastAPI Service

This document outlines the features, design choices, known issues, solutions, limitations, and assumptions of the Hindi ASR FastAPI service.

## Features

*   **ONNX Model**: Exports the NeMo Hindi Conformer-CTC model (`stt_hi_conformer_ctc_medium`) to ONNX for efficient inference. (Refers to NeMo Export Docs [2])
*   **FastAPI Endpoint**: Implements a `POST /transcribe` endpoint that accepts a WAV audio file, validates its properties (16kHz, ≤10s duration), and returns the transcribed text in JSON format. (Refers to FastAPI File Uploads [4])
*   **Async Safety**: Leverages FastAPI's `run_in_threadpool` for the CPU-bound inference call (ONNXRuntime and NeMo preprocessing) to avoid blocking the asyncio event loop. ONNXRuntime itself does not have a native `await` interface for inference. (Refers to FastAPI Concurrency Docs [5])
*   **Dockerized**: Includes a `Dockerfile` using `python:3.10-slim` as a base, installs dependencies, copies application code, and exposes port 8000 for the Uvicorn server.
*   **Example Test Case (Conceptual)**: The API is designed to transcribe short Hindi audio clips. For instance, an audio file `audio/test_hi.wav` containing "नमस्ते दुनिया" should yield the transcription "नमस्ते दुनिया". This can serve as a basic test.

## Issues & Solutions Addressed

*   **Model Download (NeMo & NGC)**:
    *   **Issue**: NeMo model downloads (`.from_pretrained()`) might require NVIDIA NGC login or credentials, or specific environment setup for accessing models.
    *   **Solution**: The `EncDecCTCModelBPE.from_pretrained()` method typically auto-downloads public models from NGC. For CI/CD or production environments with restricted internet access or private models:
        *   Manually download the `.nemo` checkpoint file.
        *   Place it in a location accessible by `export_model.py` or the application.
        *   Modify `from_pretrained()` to load from the local path or ensure NGC CLI is configured for authentication. (Refers to NeMo Model NGC Page [1])

*   **ONNX Export & Usage**:
    *   **Issue**: Exporting a NeMo model to ONNX typically yields a graph for the acoustic model but excludes the vocabulary and decoding logic. The ONNX model expects preprocessed input features (e.g., log-mel spectrograms), not raw audio.
    *   **Solution**:
        1.  The `export_model.py` script exports the acoustic model to ONNX.
        2.  In `app/model.py`, the original NeMo model (`EncDecCTCModelBPE`) is loaded again, but *only* to access its `preprocessor` (for feature extraction) and `decoder.vocabulary` (for mapping token IDs to text).
        3.  The inference pipeline is: raw audio -> NeMo preprocessor -> ONNX model inference -> CTC greedy decode using NeMo vocabulary.

*   **Preprocessing Consistency**:
    *   **Issue**: The preprocessing steps applied during inference must exactly match those used during model training to ensure accuracy.
    *   **Solution**: We use the preprocessor directly from the loaded NeMo model (`nemo_full_model.preprocessor`, which is typically an `AudioToMelSpectrogramPreprocessor` instance). This guarantees that the log-mel spectrograms fed to the ONNX model are generated in the same way as during the NeMo model's training.

*   **Async Integration with CPU-Bound Tasks**:
    *   **Issue**: FastAPI is an asynchronous framework. CPU-bound operations (like ONNX inference or extensive NumPy/PyTorch operations in preprocessing) can block the event loop if called directly in an `async def` endpoint, leading to poor performance and unresponsiveness.
    *   **Solution**: As recommended by FastAPI documentation, CPU-bound tasks are delegated to a separate thread pool using `await run_in_threadpool(func, *args)`. This allows the event loop to continue handling other requests while the blocking task runs in a worker thread. (Refers to FastAPI Concurrency Docs [5])

## Limitations & Assumptions

*   **Vocabulary & Decoding**:
    *   **Limitation**: The service uses a simple CTC greedy decoding strategy. It does not incorporate an external language model or employ more advanced beam search decoding.
    *   **Impact**: This may result in transcriptions with minor errors, especially for acoustically ambiguous phrases or out-of-vocabulary words (subword tokenization helps, but doesn't solve all OOV issues). Spacing and punctuation might not be perfect due to the nature of CTC and BPE subword concatenation.

*   **Performance**:
    *   **Observation**: ONNXRuntime inference on a CPU is generally faster than PyTorch CPU inference for the same model. However, for very high throughput or extremely low latency requirements, further optimizations might be needed.
    *   **Considerations (Beyond Current Scope)**:
        *   GPU-accelerated inference (using `onnxruntime-gpu` and `CUDAExecutionProvider`).
        *   Using NVIDIA TensorRT for further optimization on NVIDIA GPUs.
        *   Model quantization (e.g., INT8) for faster CPU/GPU inference, potentially with a slight accuracy trade-off.
        *   Batching requests on the server-side.

*   **Supported Input**:
    *   **Format**: Only WAV audio files are accepted.
    *   **Sample Rate**: Audio must be sampled at 16 kHz. (Refers to NVIDIA TAO ASR Docs [3] for similar model family requirements)
    *   **Duration**: Audio clips must be 10 seconds or shorter. Longer clips are rejected. (Users would need to segment longer audio).
    *   **Channels**: Mono audio is preferred. Stereo audio is automatically converted to mono by averaging channels.

*   **Language**:
    *   **Specificity**: The model `stt_hi_conformer_ctc_medium` is trained specifically for Hindi. (Model card typically specifies training data, e.g., ULCA Hindi dataset [6])
    *   **Impact**: It will not correctly transcribe other languages. Performance on heavily accented Hindi or code-mixed speech might vary.

---
**References (from OCR document)**:
*   [1] STT Hi Conformer-CTC Medium | NVIDIA NGC: `https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_hi_conformer_ctc_medium`
*   [2] Exporting NeMo Models - NVIDIA NeMo Framework User Guide: `https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/core/export.html`
*   [3] Speech Recognition - NVIDIA Docs (TAO Toolkit, relevant for general ASR concepts): `https://docs.nvidia.com/tao/tao-toolkit-archive/tao-30-2205/text/asr/speech_recognition.html`
*   [4] Request Files - FastAPI: `https://fastapi.tiangolo.com/tutorial/request-files/`
*   [5] Concurrency and async / await - FastAPI: `https://fastapi.tiangolo.com/async/`
*   [6] (Assumed reference for ULCA Hindi dataset, model card would confirm specific datasets)