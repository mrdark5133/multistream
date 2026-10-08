import pytest
import torch
import numpy as np
from src.ingest.embed import SigLIPEmbedder


@pytest.fixture(scope="module")
def embedder():
    return SigLIPEmbedder(model_id="google/siglip-base-patch16-224", device="cuda:0")


def test_embed_images_shape_and_norm(embedder):
    dummy_crops = [
        np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8),
        np.random.randint(0, 255, (150, 80, 3), dtype=np.uint8)
    ]
    embs = embedder.embed_images(dummy_crops, initial_batch_size=2)
    assert embs.shape == (2, 768)
    assert embs.dtype == np.float32
    
    # Check unit normalization
    norms = np.linalg.norm(embs, axis=-1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-3)


def test_embed_text_shape_and_norm(embedder):
    texts = ["a person walking", "a red car"]
    embs = embedder.embed_text(texts)
    assert embs.shape == (2, 768)
    assert embs.dtype == np.float32
    
    norms = np.linalg.norm(embs, axis=-1)
    np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-3)


def test_embed_empty(embedder):
    embs = embedder.embed_images([])
    assert embs.shape == (0, 768)
    
    embs_txt = embedder.embed_text([])
    assert embs_txt.shape == (0, 768)
