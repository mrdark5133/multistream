import numpy as np
import pytest
from src.utils.color import extract_dominant_color, match_color_query

def test_extract_dominant_color_blue():
    # Create blue BGR image (B=255, G=0, R=0)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 0] = 255
    color, conf = extract_dominant_color(img)
    assert color == "blue"
    assert conf > 0.5
    assert match_color_query("blue car", "blue") is True
    assert match_color_query("red car", "blue") is False
    assert match_color_query("a car", "blue") is None

def test_extract_dominant_color_red():
    # Create red BGR image (B=0, G=0, R=255)
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[:, :, 2] = 255
    color, conf = extract_dominant_color(img)
    assert color == "red"
    assert match_color_query("a red vehicle", "red") is True
