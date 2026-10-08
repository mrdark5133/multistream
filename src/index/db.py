import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


def get_db(db_path: str | Path) -> sqlite3.Connection:
    """Connect to SQLite database and ensure WAL mode and foreign keys."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str | Path, schema_path: Optional[str | Path] = None) -> sqlite3.Connection:
    """Initialize SQLite database with schema."""
    conn = get_db(db_path)
    if schema_path is None:
        schema_path = Path(__file__).parent / "schema.sql"
    else:
        schema_path = Path(schema_path)
        
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
    
    conn.executescript(schema_sql)
    conn.commit()
    return conn


# --- Config helpers ---

def set_config(conn: sqlite3.Connection, key: str, value: str) -> None:
    with conn:
        conn.execute(
            "INSERT INTO config (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value;",
            (key, value)
        )


def get_config(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    cur = conn.execute("SELECT value FROM config WHERE key = ?;", (key,))
    row = cur.fetchone()
    return row["value"] if row else default


# --- Embedding BLOB serialisation ---

def emb_to_blob(vec: np.ndarray) -> bytes:
    """Convert numpy float32 1D array to raw bytes."""
    vec = np.asarray(vec, dtype=np.float32)
    return vec.tobytes()


def blob_to_emb(blob: bytes) -> np.ndarray:
    """Convert raw bytes back to numpy float32 1D array."""
    return np.frombuffer(blob, dtype=np.float32)


# --- Video helpers ---

def insert_video(
    conn: sqlite3.Connection,
    path: str,
    camera: str,
    start_time: str,
    fps: float,
    width: int,
    height: int,
    duration_s: float,
    start_source: str,
    indexed_at: str,
    rotation: int = 0
) -> None:
    with conn:
        conn.execute(
            """
            INSERT INTO videos (path, camera, start_time, fps, width, height, duration_s, start_source, rotation, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                camera=excluded.camera,
                start_time=excluded.start_time,
                fps=excluded.fps,
                width=excluded.width,
                height=excluded.height,
                duration_s=excluded.duration_s,
                start_source=excluded.start_source,
                rotation=excluded.rotation,
                indexed_at=excluded.indexed_at;
            """,
            (path, camera, start_time, fps, width, height, duration_s, start_source, rotation, indexed_at)
        )


def get_video(conn: sqlite3.Connection, path: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM videos WHERE path = ?;", (path,))
    return cur.fetchone()


def list_videos(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM videos ORDER BY start_time ASC;")
    return cur.fetchall()


# --- Track helpers ---

def insert_track(
    conn: sqlite3.Connection,
    track_record: Dict[str, Any]
) -> None:
    track_record.setdefault("offset_best", track_record.get("offset_start", 0.0))
    track_record.setdefault("group", track_record.get("group_name", "object"))
    track_record.setdefault("colors", "{}")
    track_record.setdefault("quality", 0.0)
    track_record.setdefault("hits", 1)
    with conn:
        conn.execute(
            """
            INSERT INTO tracks (
                id, video, camera, track_id, label, "group", t_start, t_end, t_best,
                offset_start, offset_end, offset_best, bbox_px, bbox_norm, snapshot, emb,
                colors, quality, hits
            ) VALUES (
                :id, :video, :camera, :track_id, :label, :group, :t_start, :t_end, :t_best,
                :offset_start, :offset_end, :offset_best, :bbox_px, :bbox_norm, :snapshot, :emb,
                :colors, :quality, :hits
            ) ON CONFLICT(id) DO UPDATE SET
                label=excluded.label,
                "group"=excluded."group",
                t_start=excluded.t_start,
                t_end=excluded.t_end,
                t_best=excluded.t_best,
                offset_start=excluded.offset_start,
                offset_end=excluded.offset_end,
                offset_best=excluded.offset_best,
                bbox_px=excluded.bbox_px,
                bbox_norm=excluded.bbox_norm,
                snapshot=excluded.snapshot,
                emb=excluded.emb,
                colors=excluded.colors,
                quality=excluded.quality,
                hits=excluded.hits;
            """,
            track_record
        )


def insert_tracks_batch(
    conn: sqlite3.Connection,
    track_records: List[Dict[str, Any]]
) -> None:
    for r in track_records:
        r.setdefault("offset_best", r.get("offset_start", 0.0))
        r.setdefault("group", r.get("group_name", "object"))
        r.setdefault("colors", "{}")
        r.setdefault("quality", 0.0)
        r.setdefault("hits", 1)
    with conn:
        conn.executemany(
            """
            INSERT INTO tracks (
                id, video, camera, track_id, label, "group", t_start, t_end, t_best,
                offset_start, offset_end, offset_best, bbox_px, bbox_norm, snapshot, emb,
                colors, quality, hits
            ) VALUES (
                :id, :video, :camera, :track_id, :label, :group, :t_start, :t_end, :t_best,
                :offset_start, :offset_end, :offset_best, :bbox_px, :bbox_norm, :snapshot, :emb,
                :colors, :quality, :hits
            ) ON CONFLICT(id) DO UPDATE SET
                label=excluded.label,
                "group"=excluded."group",
                t_start=excluded.t_start,
                t_end=excluded.t_end,
                t_best=excluded.t_best,
                offset_start=excluded.offset_start,
                offset_end=excluded.offset_end,
                offset_best=excluded.offset_best,
                bbox_px=excluded.bbox_px,
                bbox_norm=excluded.bbox_norm,
                snapshot=excluded.snapshot,
                emb=excluded.emb,
                colors=excluded.colors,
                quality=excluded.quality,
                hits=excluded.hits;
            """,
            track_records
        )


# --- Frame helpers ---

def insert_frames_batch(
    conn: sqlite3.Connection,
    frame_records: List[Dict[str, Any]]
) -> None:
    with conn:
        conn.executemany(
            """
            INSERT INTO frames (id, video, camera, t_abs, offset_s, snapshot, emb)
            VALUES (:id, :video, :camera, :t_abs, :offset_s, :snapshot, :emb)
            ON CONFLICT(id) DO UPDATE SET
                t_abs=excluded.t_abs,
                offset_s=excluded.offset_s,
                snapshot=excluded.snapshot,
                emb=excluded.emb;
            """,
            frame_records
        )
