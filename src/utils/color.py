import cv2
import numpy as np
from typing import Tuple, Optional


def extract_dominant_color(crop_bgr: np.ndarray) -> Tuple[str, float]:
    """
    Extract the dominant color of an object crop by analyzing the center region
    (to ignore road, background, or asphalt boundary pixels).
    Returns: (color_name: str, confidence: float)
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return "unknown", 0.0

    h, w = crop_bgr.shape[:2]
    # Sample the center 60% of the bounding box
    y1, y2 = int(h * 0.2), int(h * 0.8)
    x1, x2 = int(w * 0.2), int(w * 0.8)

    center = crop_bgr[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else crop_bgr
    if center.size == 0:
        center = crop_bgr

    # Convert to HSV
    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    # Calculate mean statistics
    mean_v = float(np.mean(v_channel))
    mean_s = float(np.mean(s_channel))
    mean_h = float(np.mean(h_channel))

    # Dark / Black detection
    if mean_v < 48:
        conf = min(1.0, (48 - mean_v) / 30.0 + 0.6)
        return "black", round(conf, 2)

    # Low saturation: White or Gray/Silver
    if mean_s < 42:
        if mean_v > 165:
            conf = min(1.0, (mean_v - 165) / 50.0 + 0.6)
            return "white", round(conf, 2)
        else:
            return "silver/gray", 0.75

    # Chromatic hues in OpenCV [0, 180]
    if mean_h < 11 or mean_h > 168:
        return "red", 0.85
    elif 11 <= mean_h < 26:
        return "orange", 0.80
    elif 26 <= mean_h < 35:
        return "yellow", 0.85
    elif 35 <= mean_h < 85:
        return "green", 0.85
    elif 85 <= mean_h < 132:
        return "blue", 0.85
    else:
        return "purple", 0.75


COLOR_KEYWORDS = {
    "red": ["red", "maroon", "crimson", "burgundy"],
    "blue": ["blue", "navy", "cyan", "azure"],
    "green": ["green", "lime", "emerald"],
    "yellow": ["yellow", "gold"],
    "white": ["white"],
    "black": ["black", "dark"],
    "silver/gray": ["silver", "gray", "grey"],
    "orange": ["orange"],
}


def match_color_query(query_text: str, detected_color: str) -> Optional[bool]:
    """
    Check if a query specifies a color and whether it matches the detected color.
    Returns:
      True  -> query asked for this color and it matches
      False -> query asked for a DIFFERENT color
      None  -> query did not specify a color
    """
    q_lower = query_text.lower()
    specified_colors = []
    for color_name, synonyms in COLOR_KEYWORDS.items():
        for syn in synonyms:
            if f" {syn} " in f" {q_lower} " or q_lower.startswith(f"{syn} "):
                specified_colors.append(color_name)
                break

    if not specified_colors:
        return None  # No color specified in query

    return detected_color in specified_colors


CHROMATIC_SET = {"red", "blue", "green", "yellow", "orange", "brown", "purple", "pink"}


def extract_live_chromatic_color(crop_bgr: np.ndarray, min_fraction: float = 0.5) -> Tuple[Optional[str], float]:
    """
    Extract color only if its fraction is above min_fraction (default 0.5)
    and the crop is chromatic; otherwise return (None, 0.0).
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0
    try:
        from src.utils.lab_color import extract_lab_kmeans_colors
        colors = extract_lab_kmeans_colors(crop_bgr, n_clusters=3, central_crop_ratio=0.8)
        if not colors:
            return None, 0.0
        top = colors[0]
        cname = top.get("color", "").lower()
        cfrac = top.get("fraction", 0.0)
        if cfrac > min_fraction and cname in CHROMATIC_SET:
            return cname, round(cfrac, 3)
    except Exception:
        pass
    return None, 0.0
