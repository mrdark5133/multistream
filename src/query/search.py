import json
import sqlite3
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from src.index.db import get_db, blob_to_emb
from src.query.parser import QueryParser, ParsedQuery
from src.ingest.embed import SigLIPEmbedder


@dataclass
class SearchResult:
    result_id: str
    result_type: str            # 'track' | 'frame'
    camera: str
    timestamp: str              # ISO-8601
    offset_seconds: float
    score: float                # Cosine similarity [0.0, 1.0]
    label: str
    video_path: str
    snapshot_path: str
    bbox_px: Optional[List[int]] = None
    bbox_norm: Optional[List[float]] = None
    clip_path: Optional[str] = None


class SearchEngine:
    """
    MULTIStream Vector Similarity Search Engine with Spatio-Temporal Filtering & Deduplication.
    """
    def __init__(
        self,
        db_path: str | Path,
        embedder: Optional[SigLIPEmbedder] = None,
        parser: Optional[QueryParser] = None,
        device: str = "cuda:0"
    ):
        self.db_path = Path(db_path)
        self.conn = get_db(self.db_path)
        
        # Load known cameras and aliases from DB
        cur_cams = self.conn.execute("SELECT DISTINCT camera FROM videos;")
        known_cams = [r[0] for r in cur_cams.fetchall()]
        
        cur_aliases = self.conn.execute("SELECT name FROM aliases;")
        known_aliases = [r[0] for r in cur_aliases.fetchall()]
        
        self.parser = parser or QueryParser(known_cameras=known_cams, known_aliases=known_aliases)
        self.embedder = embedder or SigLIPEmbedder(device=device)

    def get_latest_timestamp(self) -> datetime.datetime:
        """Get latest indexed video start timestamp as reference 'now'."""
        cur = self.conn.execute("SELECT MAX(start_time) FROM videos;")
        row = cur.fetchone()
        if row and row[0]:
            try:
                return datetime.datetime.fromisoformat(row[0])
            except Exception:
                pass
        return datetime.datetime.now()

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        dedup_window_s: float = 5.0,
        now_override: Optional[datetime.datetime] = None
    ) -> Tuple[ParsedQuery, List[SearchResult]]:
        """
        Execute vector similarity search for natural language query.
        """
        now_ref = now_override or self.get_latest_timestamp()
        parsed = self.parser.parse(query_text, now_ref=now_ref)
        
        # 1. Embed query visual prompt
        query_emb = self.embedder.embed_text([parsed.object_prompt])[0]
        
        # 2. Query tracks table
        track_sql = "SELECT id, video, camera, track_id, label, t_start, t_end, t_best, offset_start, offset_end, offset_best, bbox_px, bbox_norm, snapshot, emb FROM tracks WHERE 1=1"
        params: List[Any] = []

        if parsed.location:
            # Check camera matching
            track_sql += " AND (LOWER(camera) LIKE ? OR LOWER(camera) LIKE ?)"
            params.extend([f"%{parsed.location.lower()}%", f"%cam_{parsed.location.lower()}%"])

        if parsed.t_start and parsed.t_end:
            track_sql += " AND (t_start <= ? AND t_end >= ?)"
            params.extend([parsed.t_end, parsed.t_start])

        cur_tracks = self.conn.execute(track_sql, params).fetchall()

        candidates: List[SearchResult] = []

        for row in cur_tracks:
            track_emb = blob_to_emb(row["emb"])
            score = float(np.dot(query_emb, track_emb))
            if similarity_threshold is None or score >= similarity_threshold:
                bbox_px = json.loads(row["bbox_px"]) if row["bbox_px"] else None
                bbox_norm = json.loads(row["bbox_norm"]) if row["bbox_norm"] else None
                candidates.append(SearchResult(
                    result_id=row["id"],
                    result_type="track",
                    camera=row["camera"],
                    timestamp=row["t_best"],
                    offset_seconds=float(row["offset_best"]),
                    score=round(score, 4),
                    label=row["label"],
                    video_path=row["video"],
                    snapshot_path=row["snapshot"],
                    bbox_px=bbox_px,
                    bbox_norm=bbox_norm
                ))

        # 3. Query whole frames table (fallback / scene context)
        frame_sql = "SELECT id, video, camera, t_abs, offset_s, snapshot, emb FROM frames WHERE 1=1"
        frame_params: List[Any] = []

        if parsed.location:
            frame_sql += " AND (LOWER(camera) LIKE ? OR LOWER(camera) LIKE ?)"
            frame_params.extend([f"%{parsed.location.lower()}%", f"%cam_{parsed.location.lower()}%"])

        if parsed.t_start and parsed.t_end:
            frame_sql += " AND (t_abs >= ? AND t_abs <= ?)"
            frame_params.extend([parsed.t_start, parsed.t_end])

        cur_frames = self.conn.execute(frame_sql, frame_params).fetchall()

        for row in cur_frames:
            f_emb = blob_to_emb(row["emb"])
            score = float(np.dot(query_emb, f_emb))
            if similarity_threshold is None or score >= similarity_threshold:
                candidates.append(SearchResult(
                    result_id=row["id"],
                    result_type="frame",
                    camera=row["camera"],
                    timestamp=row["t_abs"],
                    offset_seconds=float(row["offset_s"]),
                    score=round(score, 4),
                    label="whole_frame",
                    video_path=row["video"],
                    snapshot_path=row["snapshot"]
                ))

        # 4. Sort by score descending
        candidates.sort(key=lambda x: x.score, reverse=True)

        # 5. Temporal Deduplication on the same camera (dedup_window_s)
        deduped: List[SearchResult] = []
        for cand in candidates:
            is_dup = False
            try:
                c_dt = datetime.datetime.fromisoformat(cand.timestamp)
            except Exception:
                c_dt = None

            for existing in deduped:
                if existing.camera == cand.camera and c_dt is not None:
                    try:
                        e_dt = datetime.datetime.fromisoformat(existing.timestamp)
                        diff_s = abs((c_dt - e_dt).total_seconds())
                        if diff_s <= dedup_window_s:
                            is_dup = True
                            break
                    except Exception:
                        pass
            if not is_dup:
                deduped.append(cand)
            if len(deduped) >= top_k:
                break

        return parsed, deduped
