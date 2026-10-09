import io
import re
import logging
from pathlib import Path
from typing import Optional, List, Union, Tuple
import yaml
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

logger = logging.getLogger("multistream.voice.asr")


def build_initial_prompt(vocab_path: Optional[str | Path] = "config/vocab.yaml") -> str:
    """Build vocabulary-primed initial prompt for Whisper ASR."""
    prompt_words = [
        "person", "car", "bus", "truck", "motorcycle", "bicycle", "laptop",
        "computer monitor", "juice box", "soda can", "bluetooth speaker",
        "phone charger", "power bank", "extension board", "water bottle",
        "bottle", "umbrella", "backpack", "cell phone",
        "cam_landscape", "cam_landscape2", "test_video01", "test_video02", "test_video03",
        "mobile_cam01", "mobile_cam02", "mobile_cam03", "bounding box"
    ]
    if vocab_path:
        vp = Path(vocab_path)
        if vp.exists():
            try:
                with open(vp, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                classes = data.get("classes", [])
                for c in classes:
                    c_clean = str(c).strip().lower()
                    if c_clean and c_clean not in prompt_words:
                        prompt_words.append(c_clean)
            except Exception as e:
                logger.warning(f"[ASR] Failed reading vocab from {vocab_path}: {e}")
    return ", ".join(prompt_words) + "."


def resample_to_16k(audio_data: np.ndarray, orig_sr: int) -> np.ndarray:
    """Resample any audio array to 16,000 Hz float32."""
    if orig_sr == 16000 or len(audio_data) == 0:
        return audio_data.astype(np.float32)
    try:
        import torch
        import torchaudio
        tensor = torch.from_numpy(audio_data).unsqueeze(0)
        resampled = torchaudio.functional.resample(tensor, orig_sr, 16000)
        return resampled.squeeze(0).numpy().astype(np.float32)
    except Exception:
        num_samples = int(len(audio_data) * 16000 / orig_sr)
        indices = np.linspace(0, len(audio_data) - 1, num_samples)
        return np.interp(indices, np.arange(len(audio_data)), audio_data).astype(np.float32)


def decode_audio_bytes(raw_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode audio bytes (WAV, WebM, OGG, MP3, FLAC) to float32 mono array and sample rate."""
    with io.BytesIO(raw_bytes) as bio:
        try:
            data, sample_rate = sf.read(bio)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            return data.astype(np.float32), sample_rate
        except Exception:
            pass

    # Fallback to PyAV for WebM / Opus / AAC
    try:
        import av
        with io.BytesIO(raw_bytes) as bio:
            container = av.open(bio)
            stream = next(s for s in container.streams if s.type == "audio")
            frames = []
            sample_rate = stream.rate
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if len(arr.shape) > 1:
                    arr = arr.mean(axis=0)
                frames.append(arr)
            if frames:
                data = np.concatenate(frames)
            else:
                data = np.zeros(0, dtype=np.float32)
            if data.dtype != np.float32:
                data = data.astype(np.float32) / 32768.0
            return data, sample_rate
    except Exception as e:
        logger.error(f"[ASR] Failed decoding audio bytes: {e}")
        return np.zeros(0, dtype=np.float32), 16000


class WhisperASR:
    """
    Local CPU faster-whisper Automatic Speech Recognition.
    Runs int8 quantized Whisper models entirely on CPU.
    Uses vocabulary-primed initial_prompt for enhanced surveillance and desk object recognition.
    """
    def __init__(
        self,
        model_name: str = "base.en",
        device: str = "cpu",
        compute_type: str = "int8",
        beam_size: int = 1,
        initial_prompt: Optional[str] = None,
        vocab_path: Optional[str | Path] = "config/vocab.yaml"
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size

        # Build vocabulary-primed prompt
        if initial_prompt:
            self.initial_prompt = initial_prompt
        else:
            self.initial_prompt = build_initial_prompt(vocab_path)

        logger.info(f"[ASR] Loading faster-whisper '{self.model_name}' on {self.device} ({self.compute_type})...")
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )
        logger.info("[ASR] faster-whisper model loaded successfully.")

    @classmethod
    def from_config(cls, config_path: str | Path = "config/voice.yaml") -> "WhisperASR":
        c_path = Path(config_path)
        if c_path.exists():
            with open(c_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            asr_cfg = data.get("asr", {})
            return cls(
                model_name=asr_cfg.get("model_name", "base.en"),
                device=asr_cfg.get("device", "cpu"),
                compute_type=asr_cfg.get("compute_type", "int8"),
                beam_size=asr_cfg.get("beam_size", 1),
                initial_prompt=asr_cfg.get("initial_prompt")
            )
        return cls()

    def transcribe(
        self,
        audio_input: Union[str, Path, bytes, io.BytesIO, np.ndarray],
        use_initial_prompt: bool = True
    ) -> str:
        """
        Transcribe audio input to clean text.
        Accepts filepath, raw bytes/BytesIO, or 16kHz float32 numpy array.
        """
        prompt = self.initial_prompt if use_initial_prompt else None

        if isinstance(audio_input, np.ndarray):
            data = audio_input
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            audio_target = data.astype(np.float32)
        elif isinstance(audio_input, (bytes, io.BytesIO)):
            if isinstance(audio_input, io.BytesIO):
                raw = audio_input.getvalue()
            else:
                raw = audio_input
            data, sr = decode_audio_bytes(raw)
            data = resample_to_16k(data, sr)
            audio_target = data.astype(np.float32)
        else:
            audio_target = str(audio_input)

        segments, _ = self.model.transcribe(
            audio_target,
            beam_size=self.beam_size,
            initial_prompt=prompt,
            language="en"
        )

        texts = [s.text.strip() for s in segments if s.text.strip()]
        full_text = " ".join(texts).strip()
        # Clean double spaces
        full_text = re.sub(r"\s+", " ", full_text)
        return full_text

    def transcribe_bytes(self, audio_bytes: bytes, use_initial_prompt: bool = True):
        """Convenience method to transcribe raw audio bytes."""
        text = self.transcribe(audio_bytes, use_initial_prompt=use_initial_prompt)
        return text, {"language": "en"}
