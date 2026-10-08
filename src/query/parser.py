import re
import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from src.query.timeparse import parse_time_expression


@dataclass
class ParsedQuery:
    raw_query: str
    object_prompt: str
    location: Optional[str]
    t_start: Optional[str]        # ISO-8601 string or None
    t_end: Optional[str]          # ISO-8601 string or None
    location_status: str = "NONE" # "RESOLVED" | "UNRESOLVED" | "NONE"
    resolved_camera: Optional[str] = None
    provider: str = "rules"
    target_labels: Optional[List[str]] = None


# Synonyms mapping query words to detected YOLO-World track labels
LABEL_SYNONYMS: Dict[str, List[str]] = {
    "car": ["car", "suv", "van"],
    "cars": ["car", "suv", "van"],
    "automobile": ["car", "suv", "van"],
    "automobiles": ["car", "suv", "van"],
    "bus": ["bus"],
    "buses": ["bus"],
    "truck": ["truck", "semi-truck"],
    "trucks": ["truck", "semi-truck"],
    "semi": ["semi-truck", "truck"],
    "suv": ["suv", "car"],
    "suvs": ["suv", "car"],
    "van": ["van", "car"],
    "vans": ["van", "car"],
    "vehicle": ["car", "bus", "truck", "suv", "van", "semi-truck", "motorcycle"],
    "vehicles": ["car", "bus", "truck", "suv", "van", "semi-truck", "motorcycle"],
    "motorcycle": ["motorcycle"],
    "motorcycles": ["motorcycle"],
    "bike": ["bicycle", "motorcycle"],
    "bicycle": ["bicycle"],
    "person": ["person", "pedestrian"],
    "people": ["person", "pedestrian"],
    "pedestrian": ["pedestrian", "person"],
    "pedestrians": ["pedestrian", "person"],
    "man": ["person"],
    "woman": ["person"],
    "child": ["person"],
    "cat": ["cat"],
    "dog": ["dog"],
    "umbrella": ["umbrella"],
}


# Location preposition patterns
LOCATION_PATTERNS = [
    re.compile(r"\b(?:at|near|in|by|around|towards|through)\s+(?:the\s+)?([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)*)\b", re.IGNORECASE),
    re.compile(r"\bcamera\s+([a-zA-Z0-9_\-]+)\b", re.IGNORECASE),
    re.compile(r"\b(cam_[a-zA-Z0-9_\-]+)\b", re.IGNORECASE)
]

# Filler questions prefixes to strip
FILLER_PREFIXES = [
    re.compile(r"^(?:did\s+(?:a|any|the)|was\s+there\s+(?:a|any)|were\s+there\s+(?:any)|is\s+there\s+(?:a|any)|show\s+(?:me\s+)?(?:all\s+)?|find\s+(?:all\s+)?|search\s+(?:for\s+)?|have\s+you\s+seen\s+(?:a|any))\s+", re.IGNORECASE),
    re.compile(r"\b(?:pass(?:ed|ing)?|walk(?:ed|ing)?|enter(?:ed|ing)?|appear(?:ed|ing)?|cross(?:ed|ing)?)\b", re.IGNORECASE),
    re.compile(r"\?+$", re.IGNORECASE)
]


class QueryParser:
    """
    Offline Rule-based Natural Language Query Parser.
    Extracts visual object prompt, location referent, and temporal window without external LLM dependencies.
    """
    def __init__(self, known_cameras: Optional[List[str]] = None, known_aliases: Optional[Any] = None):
        self.known_cameras = [c.lower() for c in (known_cameras or [])]
        if isinstance(known_aliases, dict):
            self.alias_map = {k.lower(): v for k, v in known_aliases.items()}
            self.known_aliases = list(self.alias_map.keys())
        else:
            self.known_aliases = [a.lower() for a in (known_aliases or [])]
            self.alias_map = {a: a for a in self.known_aliases}

    def parse(
        self,
        query: str,
        now_ref: Optional[datetime.datetime] = None
    ) -> ParsedQuery:
        cleaned = query.strip()
        
        # 1. Parse and remove temporal expression
        start_dt, end_dt, time_span = parse_time_expression(cleaned, now_ref=now_ref)
        if time_span:
            # Remove matched time span from query string
            cleaned = re.sub(re.escape(time_span), " ", cleaned, flags=re.IGNORECASE)

        # 2. Parse and remove location referent
        location = None
        location_status = "NONE"
        resolved_camera = None

        for pat in LOCATION_PATTERNS:
            m = pat.search(cleaned)
            if m:
                cand = m.group(1).strip()
                cand_lower = cand.lower()

                # Check if matches known cameras
                matched_cam = None
                for c in self.known_cameras:
                    if cand_lower == c or cand_lower == c.replace("cam_", ""):
                        matched_cam = c
                        break

                # Check if matches known aliases
                matched_alias = None
                for a in self.known_aliases:
                    if cand_lower == a:
                        matched_alias = a
                        break

                location = cand
                cleaned = cleaned[:m.start()] + " " + cleaned[m.end():]

                if matched_cam:
                    location_status = "RESOLVED"
                    resolved_camera = matched_cam
                elif matched_alias:
                    location_status = "RESOLVED"
                    resolved_camera = self.alias_map.get(matched_alias, matched_alias)
                else:
                    location_status = "UNRESOLVED"
                    resolved_camera = None
                break

        # 3. Clean remaining text to form visual object prompt
        for pat in FILLER_PREFIXES:
            cleaned = pat.sub(" ", cleaned)

        # Remove extra whitespace and punctuation
        obj_text = re.sub(r"[^\w\s\-]", "", cleaned).strip()
        obj_text = re.sub(r"\s+", " ", obj_text).strip()

        # If empty, fallback to generic object
        if not obj_text:
            obj_text = "person or vehicle"

        # Format visual prompt with standard composite framing for SigLIP retrieval
        # E.g. "red car" -> "a red car"
        words = obj_text.split()
        if words and words[0].lower() not in ("a", "an", "the"):
            obj_prompt = f"a {obj_text}"
        else:
            obj_prompt = obj_text

        t_start_iso = start_dt.isoformat() if start_dt else None
        t_end_iso = end_dt.isoformat() if end_dt else None

        # 4. Extract target labels for category filtering
        target_labels_set = set()
        query_words = set(re.findall(r"\b[a-zA-Z]+\b", query.lower()))
        for word in query_words:
            if word in LABEL_SYNONYMS:
                target_labels_set.update(LABEL_SYNONYMS[word])
        target_labels = sorted(list(target_labels_set)) if target_labels_set else None

        return ParsedQuery(
            raw_query=query,
            object_prompt=obj_prompt,
            location=location,
            t_start=t_start_iso,
            t_end=t_end_iso,
            location_status=location_status,
            resolved_camera=resolved_camera,
            provider="rules",
            target_labels=target_labels
        )
