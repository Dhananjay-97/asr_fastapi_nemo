# File: app/model.py
import numpy as np
import onnxruntime as ort
import nemo.collections.asr as nemo_asr
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpeechRecognizer:
    def __init__(self, model_path: str, nemo_model_name: str = "stt_hi_conformer_ctc_medium"):
        logger.info(f"Initializing SpeechRecognizer with ONNX model: {model_path}")
        
        # 1. Load ONNX model for inference
        # Consider using other execution providers for performance, e.g.,
        # ['CUDAExecutionProvider', 'CPUExecutionProvider'] if GPU is available
        # or ['OpenVINOExecutionProvider', 'CPUExecutionProvider'] for Intel CPUs.
        # This requires onnxruntime-gpu or onnxruntime-openvino and respective hardware/drivers.
        try:
            self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            logger.info(f"ONNX model loaded. Input: '{self.input_name}', Output: '{self.output_name}'")
        except Exception as e:
            logger.error(f"Error loading ONNX model: {e}")
            raise

        # 2. Load NeMo model (only for its preprocessor and vocabulary)
        # This is necessary because the ONNX model expects preprocessed features.
        logger.info(f"Loading NeMo model '{nemo_model_name}' for preprocessor and vocabulary...")
        try:
            # This downloads/loads the .nemo model for its config and tokenizer
            nemo_full_model = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(model_name=nemo_model_name)
            self.preprocessor = nemo_full_model.preprocessor
            self.vocab = nemo_full_model.decoder.vocabulary # List of subword tokens
            self.blank_id = len(self.vocab) # Blank token is typically last ID in NeMo CTC models
            logger.info("NeMo preprocessor and vocabulary loaded.")
        except Exception as e:
            logger.error(f"Error loading NeMo model for preprocessor/vocab: {e}")
            raise

    def _preprocess(self, signal: np.ndarray) -> np.ndarray:
        # Ensure signal is mono float32
        if signal.ndim > 1:
            signal = signal.mean(axis=1) # Average stereo to mono
        signal = signal.astype(np.float32)

        # Convert to Torch tensor for NeMo's preprocessor
        sig_torch = torch.tensor(signal).unsqueeze(0) # Add batch dimension
        length_torch = torch.tensor([sig_torch.shape[1]]) # Length of the signal

        # Compute log-mel spectrogram using NeMo's preprocessor
        # This ensures consistency with the features used during model training.
        processed_signal, processed_length = self.preprocessor(
            input_signal=sig_torch, length=length_torch
        )
        return processed_signal.cpu().numpy()

    def _decode_greedy(self, logits: np.ndarray) -> str:
        # Perform CTC Greedy Decode
        # 1. Get the most probable token ID per time step
        pred_tokens_ids = np.argmax(logits, axis=1)

        # 2. Decode sequence
        decoded_text_parts = []
        last_token_id = -1
        for token_id in pred_tokens_ids:
            # Collapse repeats
            if token_id == last_token_id:
                continue
            last_token_id = token_id

            # Skip blank token
            if token_id == self.blank_id:
                continue
            
            # Map token ID to subword
            if 0 <= token_id < len(self.vocab):
                 decoded_text_parts.append(self.vocab[token_id])
            else:
                logger.warning(f"Warning: Encountered token_id {token_id} out of vocab range (0-{len(self.vocab)-1}), blank_id={self.blank_id}. Skipping.")


        # Join subword pieces. For BPE, simple concatenation is often sufficient.
        # More sophisticated detokenization might be needed for other tokenizers.
        return "".join(decoded_text_parts)

    def predict(self, signal: np.ndarray, sr: int) -> str:
        if sr != 16000: # Model trained on 16kHz
            # Note: Resampling should be done by the caller or in main.py for clarity
            # For now, we assume it's already 16kHz as per main.py validation.
            logger.warning(f"Input sample rate {sr}kHz is not 16kHz. Results may be suboptimal.")
            # raise ValueError("Audio must be 16 kHz.") # Or resample

        # Preprocess: Raw audio -> Log-mel spectrogram
        features = self._preprocess(signal)

        # Inference: Log-mel spectrogram -> Logits
        # The ONNX model expects a batch, [0][0] to get the logits for the single item.
        logits_batch = self.session.run([self.output_name], {self.input_name: features})
        logits = logits_batch[0][0] # [batch_size, time_steps, vocab_size] -> [time_steps, vocab_size]

        # Decode: Logits -> Text
        transcription = self._decode_greedy(logits)
        logger.info(f"Transcription: {transcription}")
        return transcription