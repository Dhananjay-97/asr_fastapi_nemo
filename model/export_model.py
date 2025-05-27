# File: model/export_model.py
import nemo.collections.asr as nemo_asr
import os

# Define the model name and output path
MODEL_NAME = "stt_hi_conformer_ctc_medium"
ONNX_OUTPUT_DIR = "model"
ONNX_OUTPUT_FILENAME = f"{MODEL_NAME}.onnx"
ONNX_OUTPUT_PATH = os.path.join(ONNX_OUTPUT_DIR, ONNX_OUTPUT_FILENAME)

def export_model():
    print(f"Loading NeMo model: {MODEL_NAME}...")
    # Load the pretrained NeMo model
    # This will download the .nemo file if not already cached by NeMo
    model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=MODEL_NAME)
    model.eval()

    # Ensure the output directory exists
    os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)

    print(f"Exporting model to ONNX: {ONNX_OUTPUT_PATH}...")
    # Export the model to ONNX format
    # Opset 13 is specified in the OCR; newer opsets might also work.
    model.export(ONNX_OUTPUT_PATH, onnx_opset_version=13)
    print(f"Model exported successfully to {ONNX_OUTPUT_PATH}")

if __name__ == "__main__":
    export_model()