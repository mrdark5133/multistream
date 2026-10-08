import cv2
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
from transformers import AutoProcessor, AutoModel

from src.utils.vram import check_vram_headroom


class SigLIPEmbedder:
    """
    SigLIP feature extractor with automatic OOM batch backoff and memory management.
    """
    def __init__(
        self,
        model_id: str = "google/siglip-base-patch16-224",
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16
    ):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype if self.device.startswith("cuda") else torch.float32

        if self.device.startswith("cuda"):
            check_vram_headroom(required_free_mib=1000)

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=self.dtype)
        self.model.to(self.device)
        self.model.eval()
        
        # Determine embedding dimension
        self.dim = self.model.config.vision_config.hidden_size

    def _extract_pooler_features(self, out: Any) -> torch.Tensor:
        """Extract feature tensor from model output."""
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        elif isinstance(out, torch.Tensor):
            return out
        else:
            raise ValueError(f"Unrecognized model output structure: {type(out)}")

    def embed_images(
        self,
        images_bgr: List[np.ndarray],
        initial_batch_size: int = 16
    ) -> np.ndarray:
        """
        Embed a list of BGR images (crops or whole frames) with automatic OOM batch backoff.
        Returns (N, D) float32 normalized embeddings array.
        """
        if not images_bgr:
            return np.zeros((0, self.dim), dtype=np.float32)

        # Convert OpenCV BGR to PIL RGB
        pil_images = []
        for img in images_bgr:
            if img is None or img.size == 0:
                # Fallback blank image
                pil_images.append(Image.new("RGB", (224, 224), (0, 0, 0)))
            else:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_images.append(Image.fromarray(rgb))

        embeddings_list = []
        batch_size = initial_batch_size
        idx = 0
        n = len(pil_images)

        while idx < n:
            current_batch = pil_images[idx : idx + batch_size]
            try:
                inputs = self.processor(images=current_batch, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    out = self.model.get_image_features(**inputs)
                    feats = self._extract_pooler_features(out)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    embeddings_list.append(feats.cpu().to(torch.float32).numpy())
                
                idx += len(current_batch)

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch_size > 1:
                    new_batch_size = max(1, batch_size // 2)
                    print(f"OOM detected in embed_images. Backing off batch size: {batch_size} -> {new_batch_size}")
                    batch_size = new_batch_size
                else:
                    # Single sample still OOMs on GPU: fallback to CPU for this batch
                    print("OOM on batch_size=1! Falling back to CPU for current sample...")
                    inputs = self.processor(images=current_batch, return_tensors="pt")
                    with torch.no_grad():
                        cpu_model = self.model.to("cpu")
                        out = cpu_model.get_image_features(**inputs)
                        feats = self._extract_pooler_features(out)
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        embeddings_list.append(feats.to(torch.float32).numpy())
                        self.model.to(self.device)
                    idx += len(current_batch)

        return np.concatenate(embeddings_list, axis=0).astype(np.float32)

    def embed_text(
        self,
        texts: List[str]
    ) -> np.ndarray:
        """
        Embed a list of text queries using standard SigLIP padding and normalization.
        Returns (N, D) float32 normalized embeddings array.
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        inputs = self.processor(
            text=texts,
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self.model.get_text_features(**inputs)
            feats = self._extract_pooler_features(out)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            return feats.cpu().to(torch.float32).numpy()
