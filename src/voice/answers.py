import datetime
import re
from typing import Dict, Any, List, Optional

CAMERA_TO_SPOKEN_NAME = {
    "cam_landscape": "landscape",
    "cam_landscape2": "landscape two",
    "test_video01": "test video one",
    "test_video02": "test video two",
    "test_video03": "test video three",
    "mobile_cam01": "mobile cam one",
    "mobile_cam02": "mobile cam two",
    "mobile_cam03": "mobile cam three",
}

FIXED_PHRASES = [
    "Searching",
    "I didn't catch that",
    "No results found",
    "Audio was empty",
    "Audio was too long",
    "Audio file was corrupted",
    "An error occurred"
]


def camera_to_spoken(cam_id: str) -> str:
    """Convert machine camera ID into natural spoken English."""
    if not cam_id:
        return "the camera"
    cid = cam_id.lower().strip()
    if cid in CAMERA_TO_SPOKEN_NAME:
        return CAMERA_TO_SPOKEN_NAME[cid]
    # Clean up prefixes
    c = re.sub(r"^cam_", "cam ", cid)
    c = c.replace("_", " ")
    return c


def format_time_for_speech(iso_ts: Optional[str], offset_s: Optional[float] = None) -> str:
    """Format timestamp into natural spoken English (e.g., 'at 3:22 AM')."""
    if iso_ts:
        try:
            dt = datetime.datetime.fromisoformat(iso_ts)
            # 12-hour format like '6:00 PM' or '3:22 AM'
            hour = dt.strftime("%I").lstrip("0")
            minute = dt.strftime("%M")
            am_pm = dt.strftime("%p")
            if minute == "00":
                return f"at {hour} {am_pm}"
            return f"at {hour}:{minute} {am_pm}"
        except Exception:
            pass
    if offset_s is not None:
        return f"at {round(offset_s, 1)} seconds"
    return ""


def template_answer(ask_response: Dict[str, Any], query_text: str = "") -> str:
    """
    Deterministically template a spoken answer from structured /ask search results.
    Strictly follows templated output (zero LLM / free-form hallucinations).
    """
    status = ask_response.get("status")

    if status == "clarify":
        referent = ask_response.get("referent", "this location")
        options: List[str] = ask_response.get("options", [])
        if options:
            spoken_options = [camera_to_spoken(opt) for opt in options[:2]]
            if len(spoken_options) == 1:
                return f"Did you mean {spoken_options[0]}?"
            return f"Did you mean {spoken_options[0]} or {spoken_options[1]}?"
        return f"Which camera covers {referent}?"

    if status == "success":
        results: List[Dict[str, Any]] = ask_response.get("results", [])
        parsed = ask_response.get("parsed", {})
        object_name = parsed.get("object_prompt") or query_text or "objects"
        n_results = len(results)
        if n_results == 0:
            return f"No objects found matching {object_name}."

        top = results[0]
        label = top.get("label", "object")
        color = top.get("color")
        camera = top.get("camera", "")
        cam_spoken = camera_to_spoken(camera)
        time_spoken = format_time_for_speech(top.get("timestamp"), top.get("offset_seconds"))
        time_part = f" {time_spoken}" if time_spoken else ""

        # Determiner and color
        if color and color not in ("unknown", "none"):
            det = "an" if color[0] in "aeiou" else "a"
            noun_phrase = f"{det} {color} {label}"
        else:
            det = "an" if label[0] in "aeiou" else "a"
            noun_phrase = f"{det} {label}"

        if n_results == 1:
            return f"Found {noun_phrase} on {cam_spoken}{time_part}."
        return f"Found {n_results} matches, most recent was {noun_phrase} on {cam_spoken}{time_part}."

    return "No results found."
