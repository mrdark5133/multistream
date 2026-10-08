import re
import json
import sqlite3
import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

from src.index.db import get_db


def normalize_alias_name(name: str) -> str:
    """
    Normalize alias strings for resilient matching across paraphrases.
    Example: 'The Main Gate!' -> 'main gate'
    """
    clean = name.strip().lower()
    # Strip leading 'the '
    if clean.startswith("the "):
        clean = clean[4:].strip()
    # Strip punctuation
    clean = re.sub(r"[^\w\s\-]", "", clean)
    # Collapse multiple whitespaces
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def point_in_polygon(x: float, y: float, polygon: List[List[float]]) -> bool:
    """
    Test whether point (x, y) lies inside a 2D polygon using ray-casting algorithm.
    Coordinates are normalized in [0.0, 1.0].
    """
    if not polygon or len(polygon) < 3:
        return True  # No polygon restriction
    
    inside = False
    n = len(polygon)
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class AliasManager:
    """
    Clarify-Once Memory Store managing human location referents and spatial polygon boundaries.
    Persists to SQLite aliases table.
    """
    def __init__(self, db_conn_or_path: sqlite3.Connection | str | Path):
        if isinstance(db_conn_or_path, (str, Path)):
            self.conn = get_db(db_conn_or_path)
            self._owns_conn = True
        else:
            self.conn = db_conn_or_path
            self._owns_conn = False

    def save_alias(
        self,
        name: str,
        camera: str,
        polygon_norm: Optional[List[List[float]]] = None
    ) -> None:
        """
        Store a human location alias mapped to a camera and optional normalized polygon.
        """
        norm_name = normalize_alias_name(name)
        if not norm_name:
            raise ValueError("Alias name cannot be empty")

        now_iso = datetime.datetime.now().isoformat()
        poly_json = json.dumps(polygon_norm) if polygon_norm else None

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO aliases (name, camera, polygon_norm, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    camera=excluded.camera,
                    polygon_norm=excluded.polygon_norm,
                    created_at=excluded.created_at;
                """,
                (norm_name, camera, poly_json, now_iso)
            )

    def resolve_alias(self, referent: str) -> Optional[Tuple[str, Optional[List[List[float]]]]]:
        """
        Resolve a referent to (camera, polygon_norm).
        Returns None if referent is unknown.
        """
        norm_ref = normalize_alias_name(referent)
        if not norm_ref:
            return None

        cur = self.conn.execute("SELECT camera, polygon_norm FROM aliases WHERE name = ?;", (norm_ref,))
        row = cur.fetchone()
        if row:
            cam = row[0]
            poly = json.loads(row[1]) if row[1] else None
            return cam, poly

        # Also check substring match if alias name is contained within referent
        cur = self.conn.execute("SELECT name, camera, polygon_norm FROM aliases;")
        for r_name, r_cam, r_poly in cur.fetchall():
            if r_name == norm_ref or r_name in norm_ref or norm_ref in r_name:
                poly = json.loads(r_poly) if r_poly else None
                return r_cam, poly

        return None

    def get_all_aliases(self) -> Dict[str, Dict[str, Any]]:
        """Get all stored aliases mapped by normalized name."""
        cur = self.conn.execute("SELECT name, camera, polygon_norm, created_at FROM aliases;")
        out = {}
        for r in cur.fetchall():
            poly = json.loads(r["polygon_norm"]) if r["polygon_norm"] else None
            out[r["name"]] = {
                "name": r["name"],
                "camera": r["camera"],
                "polygon_norm": poly,
                "created_at": r["created_at"]
            }
        return out

    def get_known_cameras(self) -> List[str]:
        """Get all distinct camera identifiers present in videos and tracks tables."""
        cur = self.conn.execute(
            """
            SELECT DISTINCT camera FROM videos
            UNION
            SELECT DISTINCT camera FROM tracks
            ORDER BY camera ASC;
            """
        )
        return [r[0] for r in cur.fetchall()]

    def delete_alias(self, name: str) -> bool:
        """Delete an alias by name."""
        norm_name = normalize_alias_name(name)
        with self.conn:
            cur = self.conn.execute("DELETE FROM aliases WHERE name = ?;", (norm_name,))
            return cur.rowcount > 0
