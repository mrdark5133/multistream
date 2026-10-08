import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

# Reference centers in OpenCV Lab color space:
# L in [0, 255], a in [0, 255], b in [0, 255]
# Neutral gray is a=128, b=128
COLOR_CENTERS = [
    # Achromatic / Neutrals
    ("black",   np.array([25.0,  128.0, 128.0])),
    ("white",   np.array([245.0, 128.0, 128.0])),
    ("grey",    np.array([128.0, 128.0, 128.0])),
    ("silver",  np.array([195.0, 128.0, 128.0])),
    
    # Chromatic
    ("red",     np.array([136.0, 208.0, 195.0])),
    ("red",     np.array([80.0,  190.0, 165.0])),   # dark red
    ("blue",    np.array([82.0,  207.0, 20.0])),
    ("blue",    np.array([40.0,  170.0, 50.0])),    # dark/navy blue
    ("blue",    np.array([140.0, 150.0, 70.0])),    # sky/light blue
    ("green",   np.array([210.0, 45.0,  205.0])),
    ("green",   np.array([100.0, 60.0,  165.0])),   # dark green
    ("yellow",  np.array([240.0, 110.0, 220.0])),
    ("orange",  np.array([175.0, 165.0, 200.0])),
    ("brown",   np.array([96.0,  154.0, 169.0])),
    ("purple",  np.array([110.0, 200.0, 85.0])),
    ("pink",    np.array([190.0, 185.0, 145.0])),
]


def classify_lab_pixel(L: float, a: float, b: float) -> str:
    """Classify a single Lab point into a named color using calibrated perceptual distance."""
    chroma = np.sqrt((a - 128.0) ** 2 + (b - 128.0) ** 2)
    
    # Very dark is always black
    if L < 45 and chroma < 35:
        return "black"
    # Very bright and low chroma is white
    if L > 225 and chroma < 25:
        return "white"
        
    # Low chroma neutrals
    if chroma < 18:
        if L < 50:
            return "black"
        elif L < 165:
            return "grey"
        elif L < 220:
            return "silver"
        else:
            return "white"

    pt = np.array([L, a, b])
    best_color = "grey"
    best_dist = float("inf")
    
    for name, center in COLOR_CENTERS:
        # Distance with lower weight on L to be robust to brightness/illumination
        diff = pt - center
        # Perceptual metric: dL*0.5, da*1.2, db*1.2
        dist = 0.5 * (diff[0] ** 2) + 1.2 * (diff[1] ** 2) + 1.2 * (diff[2] ** 2)
        if dist < best_dist:
            best_dist = dist
            best_color = name

    # Safeguard: if extremely dark, override to black
    if L < 35:
        return "black"

    return best_color


def extract_lab_kmeans_colors(
    image_bgr: np.ndarray,
    n_clusters: int = 4,
    central_crop_ratio: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Extract top 2 dominant colors with fractions using K-Means in CIE-Lab space.
    """
    if image_bgr is None or image_bgr.size == 0:
        return [{"color": "unknown", "fraction": 1.0}]

    h, w = image_bgr.shape[:2]
    if h < 4 or w < 4:
        return [{"color": "unknown", "fraction": 1.0}]

    # Optional center crop to remove background edges
    if 0.0 < central_crop_ratio < 1.0:
        ch1 = int(h * (1.0 - central_crop_ratio) / 2.0)
        ch2 = int(h * (1.0 + central_crop_ratio) / 2.0)
        cw1 = int(w * (1.0 - central_crop_ratio) / 2.0)
        cw2 = int(w * (1.0 + central_crop_ratio) / 2.0)
        roi = image_bgr[ch1:ch2, cw1:cw2]
        if roi.size > 0:
            image_bgr = roi

    # Resize small for fast k-means
    target_dim = 64
    scale = min(target_dim / max(image_bgr.shape[:2]), 1.0)
    if scale < 1.0:
        img_small = cv2.resize(image_bgr, (int(image_bgr.shape[1] * scale), int(image_bgr.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_small = image_bgr

    img_lab = cv2.cvtColor(img_small, cv2.COLOR_BGR2Lab)
    pixels = img_lab.reshape(-1, 3).astype(np.float32)

    k = min(n_clusters, len(pixels))
    if k <= 1:
        mean_lab = np.mean(pixels, axis=0)
        cname = classify_lab_pixel(mean_lab[0], mean_lab[1], mean_lab[2])
        return [{"color": cname, "fraction": 1.0}]

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()

    total_pixels = len(pixels)
    color_fractions: Dict[str, float] = {}

    for cluster_idx in range(k):
        cnt = np.sum(labels == cluster_idx)
        if cnt == 0:
            continue
        cL, ca, cb = centers[cluster_idx]
        cname = classify_lab_pixel(cL, ca, cb)
        frac = float(cnt / total_pixels)
        color_fractions[cname] = color_fractions.get(cname, 0.0) + frac

    # Sort descending by fraction
    sorted_colors = sorted(color_fractions.items(), key=lambda x: x[1], reverse=True)
    top_colors = [
        {"color": name, "fraction": round(frac, 3)}
        for name, frac in sorted_colors[:2]
    ]

    return top_colors if top_colors else [{"color": "unknown", "fraction": 1.0}]


def extract_track_colors(
    crop_bgr: np.ndarray,
    label: str,
    group: str
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Extract color dictionary based on object group:
    - person: {"person_upper": [...], "person_lower": [...]}
    - vehicle: {"vehicle": [...]}
    - object: {"object": [...]}
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return {}

    h, w = crop_bgr.shape[:2]

    if group == "person" or label.lower() in ("person", "pedestrian", "cyclist", "child"):
        # Split vertically into upper and lower body
        mid_y = max(1, int(h * 0.5))
        upper_crop = crop_bgr[0:mid_y, :]
        lower_crop = crop_bgr[mid_y:h, :]

        upper_colors = extract_lab_kmeans_colors(upper_crop, n_clusters=3, central_crop_ratio=0.85)
        lower_colors = extract_lab_kmeans_colors(lower_crop, n_clusters=3, central_crop_ratio=0.85)
        return {
            "person_upper": upper_colors,
            "person_lower": lower_colors
        }
    elif group == "vehicle" or label.lower() in ("car", "sedan", "suv", "van", "truck", "bus", "motorcycle", "bicycle"):
        veh_colors = extract_lab_kmeans_colors(crop_bgr, n_clusters=4, central_crop_ratio=0.8)
        return {
            "vehicle": veh_colors
        }
    else:
        obj_colors = extract_lab_kmeans_colors(crop_bgr, n_clusters=3, central_crop_ratio=0.85)
        return {
            "object": obj_colors
        }
