import re
from typing import List, Optional, Tuple, Dict
from rapidfuzz import fuzz, process

CAMERA_SPOKEN_MAP: Dict[str, List[str]] = {
    "cam_landscape": [
        "landscape", "cam landscape", "camera landscape", "landscape camera",
        "landscape one", "landscape 1", "cam landscape 1", "cam landscape one"
    ],
    "cam_landscape2": [
        "landscape two", "landscape 2", "cam landscape two", "cam landscape 2",
        "landscape two camera", "landscape 2 camera", "second landscape"
    ],
    "test_video01": [
        "test video one", "test video 1", "video one", "video 1",
        "test video", "first test video", "test video 01"
    ],
    "test_video02": [
        "test video two", "test video 2", "video two", "video 2",
        "second test video", "test video 02"
    ],
    "test_video03": [
        "test video three", "test video 3", "video three", "video 3",
        "third test video", "test video 03"
    ],
    "mobile_cam01": [
        "mobile cam one", "mobile cam 1", "mobile camera one", "mobile 1", "mobile one",
        "phone camera", "mobile cam 01"
    ],
    "mobile_cam02": [
        "mobile cam two", "mobile cam 2", "mobile camera two", "mobile 2", "mobile two",
        "mobile cam 02"
    ],
    "mobile_cam03": [
        "mobile cam three", "mobile cam 3", "mobile camera three", "mobile 3", "mobile three",
        "mobile cam 03"
    ]
}


def clean_spoken_text(text: str) -> str:
    """Lowercase and strip punctuation."""
    t = text.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def match_spoken_to_camera(spoken_input: str, candidates: Optional[List[str]] = None, threshold: float = 65.0) -> Optional[str]:
    """
    Fuzzy-matches a spoken phrase (e.g. 'landscape two' or 'say landscape two')
    to one of the camera candidates using rapidfuzz.
    """
    cleaned = clean_spoken_text(spoken_input)
    # Strip common filler prefixes
    for prefix in ("say ", "it is ", "it s ", "camera ", "the ", "at "):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()

    valid_cams = set(candidates) if candidates else set(CAMERA_SPOKEN_MAP.keys())

    # Build target mapping (spoken_alias -> actual_camera_id)
    choices: Dict[str, str] = {}
    for cam_id in valid_cams:
        # Include raw camera ID and cleaned version
        choices[cam_id.lower()] = cam_id
        choices[clean_spoken_text(cam_id)] = cam_id
        for alias in CAMERA_SPOKEN_MAP.get(cam_id, []):
            choices[alias] = cam_id

    # Try exact match first
    if cleaned in choices:
        return choices[cleaned]

    # Fuzzy match with token_sort_ratio
    result = process.extractOne(cleaned, list(choices.keys()), scorer=fuzz.token_sort_ratio)
    if result:
        matched_alias, score, _ = result
        if score >= threshold:
            return choices[matched_alias]

    # Fallback to partial ratio
    result_partial = process.extractOne(cleaned, list(choices.keys()), scorer=fuzz.partial_ratio)
    if result_partial:
        matched_alias, score, _ = result_partial
        if score >= threshold + 10:
            return choices[matched_alias]

    return None


def normalize_query_cameras(query_text: str) -> str:
    """
    Fuzzy-matches and normalizes camera mentions inside natural language queries
    e.g. 'at the landscape two camera' -> 'at cam_landscape2'
         'near landscape two' -> 'near cam_landscape2'
         'at test video one' -> 'at test_video01'
    """
    normalized = query_text

    patterns = [
        # (regex pattern, replacement canonical camera name)
        (r"\b(?:at|near|in|by|around)?\s*(?:the\s+)?(?:cam\s+)?landscape\s*(?:two|2)(?:\s+camera)?\b", "at cam_landscape2"),
        (r"\b(?:at|near|in|by|around)?\s*(?:the\s+)?(?:cam\s+)?landscape(?:\s+camera|\s+(?:one|1))?\b", "at cam_landscape"),
        (r"\b(?:at|near|in|by|around)?\s*(?:the\s+)?test\s+video\s*(?:one|1|01)\b", "at test_video01"),
        (r"\b(?:at|near|in|by|around)?\s*(?:the\s+)?test\s+video\s*(?:two|2|02)\b", "at test_video02"),
        (r"\b(?:at|near|in|by|around)?\s*(?:the\s+)?test\s+video\s*(?:three|3|03)\b", "at test_video03"),
        (r"\b(?:at|near|in|by|around)?\s*(?:the\s+)?mobile\s+cam\s*(?:one|1|01)\b", "at mobile_cam01"),
        (r"\b(?:at|near|in|by|around)?\s*(?:the\s+)?mobile\s+cam\s*(?:two|2|02)\b", "at mobile_cam02"),
        (r"\b(?:at|near|in|by|around)?\s*(?:the\s+)?mobile\s+cam\s*(?:three|3|03)\b", "at mobile_cam03"),
    ]

    for pat, repl in patterns:
        normalized = re.sub(pat, repl, normalized, flags=re.IGNORECASE)

    # Clean double 'at at' if regex substituted
    normalized = re.sub(r"\b(?:at|in|near)\s+(at|in|near)\b", r"\1", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized
