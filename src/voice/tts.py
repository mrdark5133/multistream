import os
import wave
import hashlib
import re
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple
import yaml

from piper import PiperVoice
from src.voice.answers import FIXED_PHRASES

logger = logging.getLogger("multistream.voice.tts")


class PiperTTS:
    """
    Local CPU Piper Text-to-Speech Engine.
    Uses 'en_US-lessac-low' ONNX model on CPU.
    Caches all spoken phrases on disk and pre-synthesizes fixed system prompts.
    """
    def __init__(
        self,
        model_path: str | Path = "weights/piper/en/en_US/lessac/low/en_US-lessac-low.onnx",
        config_path: str | Path = "weights/piper/en/en_US/lessac/low/en_US-lessac-low.onnx.json",
        cache_dir: str | Path = "cache/audio",
        use_cuda: bool = False
    ):
        self.model_path = Path(model_path)
        self.config_path = Path(config_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.use_cuda = use_cuda

        if not self.model_path.exists():
            raise FileNotFoundError(f"Piper model not found at: {self.model_path}")
        if not self.config_path.exists():
            raise FileNotFoundError(f"Piper config not found at: {self.config_path}")

        logger.info(f"[TTS] Loading PiperVoice from {self.model_path} on CPU...")
        self.voice = PiperVoice.load(
            str(self.model_path),
            config_path=str(self.config_path),
            use_cuda=self.use_cuda
        )
        logger.info("[TTS] PiperVoice loaded successfully.")

        self._fixed_cache: Dict[str, str] = {}
        self.precompute_fixed_phrases()

    @classmethod
    def from_config(cls, config_path: str | Path = "config/voice.yaml") -> "PiperTTS":
        c_path = Path(config_path)
        if c_path.exists():
            with open(c_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            tts_cfg = data.get("tts", {})
            return cls(
                model_path=tts_cfg.get("model_path", "weights/piper/en/en_US/lessac/low/en_US-lessac-low.onnx"),
                config_path=tts_cfg.get("config_path", "weights/piper/en/en_US/lessac/low/en_US-lessac-low.onnx.json"),
                cache_dir=tts_cfg.get("cache_dir", "cache/audio"),
                use_cuda=tts_cfg.get("use_cuda", False)
            )
        return cls()

    @staticmethod
    def _text_to_id(text: str) -> str:
        clean = text.strip().lower()
        return hashlib.md5(clean.encode("utf-8")).hexdigest()[:16]

    def precompute_fixed_phrases(self) -> None:
        """Pre-synthesize fixed phrases on startup so they play with zero generation latency."""
        for phrase in FIXED_PHRASES:
            audio_id, _ = self.synthesize(phrase)
            self._fixed_cache[phrase.strip().lower()] = audio_id
        logger.info(f"[TTS] Pre-synthesized {len(FIXED_PHRASES)} fixed system phrases.")

    def synthesize(self, text: str) -> Tuple[str, Path]:
        """
        Synthesize text to 16/22kHz WAV on disk.
        Returns (audio_id, file_path).
        """
        text_clean = text.strip()
        if not text_clean:
            text_clean = "I didn't catch that."

        audio_id = self._text_to_id(text_clean)
        out_path = self.cache_dir / f"{audio_id}.wav"

        # If already cached on disk, return immediately
        if out_path.exists() and out_path.stat().st_size > 0:
            return audio_id, out_path

        # Synthesize using Piper
        try:
            with wave.open(str(out_path), "wb") as wav_file:
                self.voice.synthesize_wav(text_clean, wav_file)
        except Exception as e:
            logger.error(f"[TTS] Error synthesizing text '{text_clean}': {e}")
            raise e

        return audio_id, out_path

    def get_audio_path(self, audio_id: str) -> Optional[Path]:
        """Retrieve cached audio file by ID."""
        clean_id = re.sub(r"[^\w\-]", "", audio_id)
        candidate = self.cache_dir / f"{clean_id}.wav"
        if candidate.exists() and candidate.is_file():
            return candidate
        return None
