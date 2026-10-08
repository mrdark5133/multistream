import json
import sqlite3
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import cv2
from src.index.db import get_db, blob_to_emb
from src.query.parser import QueryParser, ParsedQuery
from src.ingest.embed import SigLIPEmbedder
from src.memory.aliases import AliasManager, point_in_polygon
from src.query.rerank import VLMReranker
from src.utils.color import extract_dominant_color, match_color_query


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
    color: Optional[str] = None
    bbox_px: Optional[List[int]] = None
    bbox_norm: Optional[List[float]] = None
    clip_path: Optional[str] = None


class SearchEngine:
    """
    MULTIStream Vector Similarity Search Engine with Spatio-Temporal Filtering,
    Clarify-Once Memory Integration, and Temporal Deduplication.
    """
    def __init__(
        self,
        db_path: str | Path,
        embedder: Optional[SigLIPEmbedder] = None,
        parser: Optional[QueryParser] = None,
        reranker: Optional[VLMReranker] = None,
        device: str = "cuda:0"
    ):
        self.db_path = Path(db_path)
        self.conn = get_db(self.db_path)
        self.alias_mgr = AliasManager(self.conn)
        self.reload_parser(parser=parser)
        self.embedder = embedder or SigLIPEmbedder(device=device)
        self.reranker = reranker or VLMReranker()

    def reload_parser(self, parser: Optional[QueryParser] = None) -> None:
        """Reload known cameras and saved aliases into query parser."""
        known_cams = self.alias_mgr.get_known_cameras()
        aliases_map = self.alias_mgr.get_all_aliases()
        known_alias_dict = {name: data["camera"] for name, data in aliases_map.items()}
        self.parser = parser or QueryParser(known_cameras=known_cams, known_aliases=known_alias_dict)

    def save_alias(
        self,
        name: str,
        camera: str,
        polygon_norm: Optional[List[List[float]]] = None
    ) -> None:
        """Save a human location referent and reload parser memory."""
        self.alias_mgr.save_alias(name, camera, polygon_norm=polygon_norm)
        self.reload_parser()

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

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        dedup_window_s: float = 5.0,
        now_override: Optional[datetime.datetime] = None
    ) -> Dict[str, Any]:
        """
        Unified Query function with clarify-once protocol.
        If location is unknown, returns {"status": "clarify", "referent": referent, "options": known_cameras}.
        If location is known or resolved, returns {"status": "success", "parsed": parsed, "results": results}.
        """
        now_ref = now_override or self.get_latest_timestamp()
        parsed = self.parser.parse(query_text, now_ref=now_ref)
        if parsed.location and parsed.location_status == "UNRESOLVED":
            self.reload_parser()
            parsed = self.parser.parse(query_text, now_ref=now_ref)
        
        # Check if query contains an unknown location referent
        if parsed.location:
            is_known_cam = any(
                parsed.location.lower() == cam.lower() or parsed.location.lower() == cam.lower().replace("cam_", "")
                for cam in self.alias_mgr.get_known_cameras()
            )
            is_known_alias = (self.alias_mgr.resolve_alias(parsed.location) is not None)
            
            if not is_known_cam and not is_known_alias:
                # Trigger clarify-once request
                return {
                    "status": "clarify",
                    "referent": parsed.location,
                    "options": self.alias_mgr.get_known_cameras(),
                    "parsed": parsed
                }

        parsed, results = self.search(
            query_text=query_text,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            dedup_window_s=dedup_window_s,
            now_override=now_override
        )
        return {
            "status": "success",
            "parsed": parsed,
            "results": results
        }

    def search(
        self,
        query_text: str,
        top_k: int = 10,
        similarity_threshold: Optional[float] = None,
        dedup_window_s: float = 5.0,
        now_override: Optional[datetime.datetime] = None
    ) -> Tuple[ParsedQuery, List[SearchResult]]:
        """
        Execute vector similarity search for natural language query with spatial & temporal filtering.
        """
        now_ref = now_override or self.get_latest_timestamp()
        parsed = self.parser.parse(query_text, now_ref=now_ref)
        
        # Check alias resolution and polygon filter
        target_camera = None
        target_polygon = None
        
        if parsed.location:
            # 1. Check known cameras
            for cam in self.alias_mgr.get_known_cameras():
                if parsed.location.lower() == cam.lower() or parsed.location.lower() == cam.lower().replace("cam_", ""):
                    target_camera = cam
                    break
            
            # 2. Check saved aliases
            if not target_camera:
                res_alias = self.alias_mgr.resolve_alias(parsed.location)
                if res_alias:
                    target_camera, target_polygon = res_alias

        # 1. Embed query visual prompt
        query_emb = self.embedder.embed_text([parsed.object_prompt])[0]
        
        # 2. Query tracks table dynamically supporting new Phase 1b schema columns
        track_cols = {c[1] for c in self.conn.execute("PRAGMA table_info(tracks);").fetchall()}
        cols_to_select = ["id", "video", "camera", "track_id", "label", "t_start", "t_end", "t_best",
                          "offset_start", "offset_end", "offset_best", "bbox_px", "bbox_norm", "snapshot", "emb"]
        if "colors" in track_cols:
            cols_to_select.append("colors")
        if "quality" in track_cols:
            cols_to_select.append("quality")
        if "hits" in track_cols:
            cols_to_select.append("hits")
        if "group" in track_cols:
            cols_to_select.append('"group"')
        track_sql = f"SELECT {', '.join(cols_to_select)} FROM tracks WHERE 1=1"
        params: List[Any] = []

        if target_camera:
            cam_norm = target_camera.lower()
            cam_bare = cam_norm.replace("cam_", "")
            cam_with = f"cam_{cam_bare}"
            track_sql += " AND LOWER(camera) IN (?, ?, ?)"
            params.extend([cam_norm, cam_bare, cam_with])

        if parsed.t_start and parsed.t_end:
            track_sql += " AND (t_start <= ? AND t_end >= ?)"
            params.extend([parsed.t_end, parsed.t_start])

        cur_tracks = self.conn.execute(track_sql, params).fetchall()

        candidates: List[SearchResult] = []

        for row in cur_tracks:
            # Enforce target category filtering (e.g. asking for car/bus ignores person)
            if parsed.target_labels:
                track_label = row["label"].lower()
                if track_label not in parsed.target_labels:
                    continue

            bbox_norm = json.loads(row["bbox_norm"]) if row["bbox_norm"] else None
            # Apply spatial polygon filter if defined for this alias
            if target_polygon and bbox_norm and len(bbox_norm) == 4:
                cx = (bbox_norm[0] + bbox_norm[2]) / 2.0
                cy = (bbox_norm[1] + bbox_norm[3]) / 2.0
                if not point_in_polygon(cx, cy, target_polygon):
                    continue

            track_emb = blob_to_emb(row["emb"])
            score = float(np.dot(query_emb, track_emb))

            # Color attribute extraction & soft boost w=0.03 matching
            det_color = "unknown"
            track_colors_list: List[str] = []
            
            # Check new colors JSON column first
            if "colors" in row.keys() and row["colors"]:
                try:
                    cdata = json.loads(row["colors"])
                    for part, clist in cdata.items():
                        if isinstance(clist, list):
                            for item in clist:
                                cname = item.get("color", "")
                                if cname and cname != "unknown":
                                    track_colors_list.append(cname.lower())
                    if track_colors_list:
                        det_color = track_colors_list[0]
                except Exception:
                    pass

            snap_p = Path(row["snapshot"])
            if not snap_p.exists():
                snap_p = self.db_path.parent / row["snapshot"]

            # Fallback to legacy extraction if no colors column in DB
            if not track_colors_list and snap_p.exists() and bbox_norm:
                img = cv2.imread(str(snap_p))
                if img is not None:
                    h, w = img.shape[:2]
                    bx1, by1 = max(0, int(bbox_norm[0] * w)), max(0, int(bbox_norm[1] * h))
                    bx2, by2 = min(w, int(bbox_norm[2] * w)), min(h, int(bbox_norm[3] * h))
                    crop = img[by1:by2, bx1:bx2]
                    if crop.size > 0:
                        det_color, _ = extract_dominant_color(crop)
                        if det_color != "unknown":
                            track_colors_list.append(det_color.lower())

            # Check if query specified a color for soft boost (w=0.03)
            query_words = set(query_text.lower().split())
            standard_colors = {"red", "blue", "green", "yellow", "black", "white", "silver", "grey", "orange", "brown", "purple", "pink"}
            matched_query_colors = query_words.intersection(standard_colors)

            if matched_query_colors:
                if any(qc in track_colors_list for qc in matched_query_colors):
                    score += 0.03  # soft boost w=0.03
                elif track_colors_list:
                    score -= 0.02  # soft mismatch penalty

            if similarity_threshold is None or score >= similarity_threshold:
                bbox_px = json.loads(row["bbox_px"]) if row["bbox_px"] else None
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
                    color=det_color,
                    bbox_px=bbox_px,
                    bbox_norm=bbox_norm
                ))

        # 3. Query whole frames table (fallback / scene context, skipped if polygon filter is active or target tracks found)
        if not target_polygon and (not parsed.target_labels or len(candidates) == 0):
            frame_sql = "SELECT id, video, camera, t_abs, offset_s, snapshot, emb FROM frames WHERE 1=1"
            frame_params: List[Any] = []

            if target_camera:
                cam_norm = target_camera.lower()
                cam_bare = cam_norm.replace("cam_", "")
                cam_with = f"cam_{cam_bare}"
                frame_sql += " AND LOWER(camera) IN (?, ?, ?)"
                frame_params.extend([cam_norm, cam_bare, cam_with])

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

        # 6. Optional VLM Reranking (Phase 6)
        if self.reranker and self.reranker.provider != "off":
            deduped, _ = self.reranker.rerank(parsed.object_prompt, deduped, top_k=top_k)

        return parsed, deduped


def save_alias(
    name: str,
    camera: str,
    polygon_norm: Optional[List[List[float]]] = None,
    db_path: str | Path = "index_base/index.db"
) -> None:
    """Module-level save_alias helper."""
    mgr = AliasManager(db_path)
    mgr.save_alias(name, camera, polygon_norm=polygon_norm)


def query(
    query_text: str,
    db_path: str | Path = "index_base/index.db",
    engine: Optional[SearchEngine] = None,
    **kwargs
) -> Dict[str, Any]:
    """Module-level query helper with clarify-once protocol."""
    eng = engine or SearchEngine(db_path=db_path)
    return eng.query(query_text, **kwargs)
