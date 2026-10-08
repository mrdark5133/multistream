import sqlite3
import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

from src.index.db import blob_to_emb, get_db


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized or unnormalized 1D vectors."""
    a_flat = np.asarray(a, dtype=np.float32).flatten()
    b_flat = np.asarray(b, dtype=np.float32).flatten()
    dot = float(np.dot(a_flat, b_flat))
    norm_a = float(np.linalg.norm(a_flat))
    norm_b = float(np.linalg.norm(b_flat))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


def find_cross_camera_matches(
    conn: sqlite3.Connection,
    query_track_id: str,
    min_similarity: float = 0.60,
    max_time_gap_s: float = 7200.0,
    same_camera_allowed: bool = False
) -> Dict[str, Any]:
    """
    Find matching tracks for a given track across other cameras.
    Enforces cross-camera constraint and temporal proximity.
    """
    cur = conn.execute("SELECT * FROM tracks WHERE id = ?;", (query_track_id,))
    query_row = cur.fetchone()
    if not query_row:
        raise ValueError(f"Track ID '{query_track_id}' not found in database.")

    q_cam = query_row["camera"]
    q_emb = blob_to_emb(query_row["emb"])
    q_t_start = query_row["t_start"]
    q_dt = datetime.datetime.fromisoformat(q_t_start)

    # Query candidate tracks
    if same_camera_allowed:
        candidate_cur = conn.execute("SELECT * FROM tracks WHERE id != ?;", (query_track_id,))
    else:
        candidate_cur = conn.execute("SELECT * FROM tracks WHERE camera != ?;", (q_cam,))

    candidates = candidate_cur.fetchall()
    matches = []

    for c in candidates:
        c_dt = datetime.datetime.fromisoformat(c["t_start"])
        time_diff_s = (c_dt - q_dt).total_seconds()

        # Enforce temporal window
        if abs(time_diff_s) > max_time_gap_s:
            continue

        c_emb = blob_to_emb(c["emb"])
        sim = cosine_similarity(q_emb, c_emb)

        if sim >= min_similarity:
            matches.append({
                "track_id": c["id"],
                "camera": c["camera"],
                "label": c["label"],
                "similarity": round(sim, 4),
                "t_start": c["t_start"],
                "t_end": c["t_end"],
                "offset_best": c["offset_best"],
                "snapshot": c["snapshot"],
                "time_delta_s": round(time_diff_s, 2),
                "chronological_order": "after" if time_diff_s > 0 else "before" if time_diff_s < 0 else "concurrent"
            })

    # Sort matches by chronological time, then similarity
    matches.sort(key=lambda m: (m["t_start"], -m["similarity"]))

    # Generate trajectory narrative
    trajectory_steps = [f"[{query_row['camera']} @ {query_row['t_start']}]"]
    for m in matches:
        trajectory_steps.append(f"-> [{m['camera']} @ {m['t_start']} (sim: {m['similarity']:.2f})]")
    trajectory_summary = " ".join(trajectory_steps)

    return {
        "query_track": {
            "id": query_row["id"],
            "camera": query_row["camera"],
            "label": query_row["label"],
            "t_start": query_row["t_start"],
            "t_end": query_row["t_end"],
            "snapshot": query_row["snapshot"]
        },
        "matches": matches,
        "trajectory_summary": trajectory_summary
    }


def find_all_cross_camera_associations(
    conn: sqlite3.Connection,
    min_similarity: float = 0.65,
    max_time_gap_s: float = 7200.0,
    top_k_per_track: int = 3
) -> List[Dict[str, Any]]:
    """
    Scan all tracks in the database and detect likely cross-camera re-identifications.
    """
    cur = conn.execute("SELECT * FROM tracks ORDER BY t_start ASC;")
    all_tracks = cur.fetchall()

    associations = []
    seen_pairs = set()

    for row in all_tracks:
        res = find_cross_camera_matches(
            conn,
            query_track_id=row["id"],
            min_similarity=min_similarity,
            max_time_gap_s=max_time_gap_s,
            same_camera_allowed=False
        )
        for m in res["matches"][:top_k_per_track]:
            pair_key = tuple(sorted([row["id"], m["track_id"]]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            associations.append({
                "track_a": row["id"],
                "camera_a": row["camera"],
                "time_a": row["t_start"],
                "label_a": row["label"],
                "snapshot_a": row["snapshot"],
                "track_b": m["track_id"],
                "camera_b": m["camera"],
                "time_b": m["t_start"],
                "label_b": m["label"],
                "snapshot_b": m["snapshot"],
                "similarity": m["similarity"],
                "time_delta_s": m["time_delta_s"],
                "path": f"{row['camera']} ({row['t_start']}) -> {m['camera']} ({m['t_start']})"
            })

    associations.sort(key=lambda a: -a["similarity"])
    return associations
