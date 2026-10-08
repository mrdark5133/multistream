import os
import json
import sqlite3
import datetime
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.query.search import SearchEngine
from src.query.clips import extract_video_clip
from src.utils.vram import get_nvml_vram_info


# Project directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = ROOT_DIR / "static"
INDEX_DB = ROOT_DIR / "index_base" / "index.db"
CLIPS_DIR = ROOT_DIR / "clips"
SNAPSHOTS_DIR = ROOT_DIR / "snapshots"
UPLOADS_DIR = ROOT_DIR / "footage" / "uploads"

# Ensure directories exist
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="MULTIStream Video Search API",
    version="1.0.0",
    description="Natural language video retrieval, clarify-once spatial memory, and on-demand clip extraction."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory search engine and session store
_engine: Optional[SearchEngine] = None
session_history: Dict[str, List[Dict[str, Any]]] = {}


def get_search_engine() -> SearchEngine:
    global _engine
    if _engine is None:
        if not INDEX_DB.exists():
            raise RuntimeError(f"Database not found at {INDEX_DB}. Please run ingest first.")
        _engine = SearchEngine(db_path=INDEX_DB)
    return _engine


# --- Request Models ---

class AskRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    session_id: str = Field("default", description="Client session identifier")
    top_k: int = Field(5, description="Number of results to return")
    now: Optional[str] = Field(None, description="ISO-8601 override for reference now")


class AliasRequest(BaseModel):
    name: str = Field(..., description="Human location name to register")
    camera: str = Field(..., description="Camera identifier")
    polygon: Optional[List[List[float]]] = Field(None, description="Normalized 2D polygon vertices [[x,y],...]")


# --- Endpoints ---

@app.get("/")
def serve_root():
    """Serve the static chat UI."""
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html, media_type="text/html")
    return {"message": "MULTIStream API is active. UI not found in static/index.html"}


@app.post("/ask")
def ask(req: AskRequest):
    """
    Execute natural language query with clarify-once protocol.
    Returns status='clarify' when referent is unknown, or status='success' with ranked results.
    """
    engine = get_search_engine()
    now_override = datetime.datetime.fromisoformat(req.now) if req.now else None

    resp = engine.query(
        query_text=req.query,
        top_k=req.top_k,
        now_override=now_override
    )

    # Multi-turn session history (keep last 5 turns)
    if req.session_id not in session_history:
        session_history[req.session_id] = []
    session_history[req.session_id].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "query": req.query,
        "status": resp["status"]
    })
    session_history[req.session_id] = session_history[req.session_id][-5:]

    if resp["status"] == "clarify":
        return {
            "status": "clarify",
            "referent": resp["referent"],
            "options": resp["options"]
        }

    parsed = resp["parsed"]
    formatted_results = []
    for r in resp["results"]:
        formatted_results.append({
            "id": r.result_id,
            "type": r.result_type,
            "camera": r.camera,
            "timestamp": r.timestamp,
            "offset_seconds": r.offset_seconds,
            "score": r.score,
            "label": r.label,
            "snapshot_url": f"/snapshot/{r.result_id}",
            "clip_url": f"/clip/{r.result_id}",
            "bbox_px": r.bbox_px,
            "bbox_norm": r.bbox_norm
        })

    return {
        "status": "success",
        "query": req.query,
        "parsed": {
            "object_prompt": parsed.object_prompt,
            "location": parsed.location,
            "location_status": parsed.location_status,
            "resolved_camera": parsed.resolved_camera,
            "t_start": parsed.t_start,
            "t_end": parsed.t_end,
            "provider": parsed.provider
        },
        "results": formatted_results
    }


@app.post("/alias")
def save_alias_endpoint(req: AliasRequest):
    """
    Register a human location alias with camera mapping and optional polygon boundary.
    """
    engine = get_search_engine()
    engine.save_alias(
        name=req.name,
        camera=req.camera,
        polygon_norm=req.polygon
    )
    return {
        "status": "saved",
        "name": req.name,
        "camera": req.camera,
        "polygon": req.polygon
    }


@app.get("/cameras")
def get_cameras():
    """
    List known cameras with frame resolution, start time, and duration.
    """
    engine = get_search_engine()
    cur = engine.conn.execute(
        """
        SELECT camera, path, start_time, fps, width, height, duration_s, rotation
        FROM videos
        GROUP BY camera
        ORDER BY camera ASC;
        """
    )
    cameras = []
    for r in cur.fetchall():
        cameras.append({
            "camera": r["camera"],
            "sample_path": r["path"],
            "start_time": r["start_time"],
            "fps": r["fps"],
            "width": r["width"],
            "height": r["height"],
            "duration_s": r["duration_s"],
            "rotation": r["rotation"]
        })
    return cameras


@app.get("/aliases")
def get_aliases():
    """
    List all stored aliases in the memory table.
    """
    engine = get_search_engine()
    return engine.alias_mgr.get_all_aliases()


@app.get("/snapshot/{item_id:path}")
def get_snapshot(item_id: str):
    """
    Serve snapshot JPEG for a track, frame, direct path, or camera preview.
    """
    engine = get_search_engine()
    
    # 1. Direct file path checks
    candidate_bases = [
        ROOT_DIR / "index_base",
        ROOT_DIR,
        ROOT_DIR / "index_base" / "snapshots",
        ROOT_DIR / "snapshots"
    ]
    for base in candidate_bases:
        p = base / item_id
        if p.exists() and p.is_file():
            return FileResponse(p, media_type="image/jpeg")

    # 2. Check track or frame ID in DB
    cur = engine.conn.execute("SELECT snapshot FROM tracks WHERE id = ?;", (item_id,))
    row = cur.fetchone()
    if not row:
        cur = engine.conn.execute("SELECT snapshot FROM frames WHERE id = ?;", (item_id,))
        row = cur.fetchone()

    # 3. If item_id is a camera name, find first snapshot for that camera
    if not row:
        cur = engine.conn.execute(
            "SELECT snapshot FROM tracks WHERE camera = ? OR camera = ? LIMIT 1;",
            (item_id, f"cam_{item_id}")
        )
        row = cur.fetchone()

    if row and row[0]:
        snap_rel = row[0]
        for base in [ROOT_DIR / "index_base", ROOT_DIR]:
            candidate = base / snap_rel
            if candidate.exists() and candidate.is_file():
                return FileResponse(candidate, media_type="image/jpeg")

    raise HTTPException(status_code=404, detail=f"Snapshot not found for item '{item_id}'")


@app.get("/clip/{result_id}")
def get_clip(result_id: str):
    """
    Extract on-demand video clip with ffmpeg and serve mp4.
    """
    engine = get_search_engine()
    cur = engine.conn.execute(
        "SELECT video, offset_best, offset_start FROM tracks WHERE id = ?;",
        (result_id,)
    )
    row = cur.fetchone()
    if not row:
        cur = engine.conn.execute(
            "SELECT video, offset_s FROM frames WHERE id = ?;",
            (result_id,)
        )
        row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Result '{result_id}' not found in index.")
        v_path = ROOT_DIR / row["video"]
        offset_s = float(row["offset_s"])
    else:
        v_path = ROOT_DIR / row["video"]
        offset_s = float(row["offset_best"] if "offset_best" in row.keys() else row["offset_start"])

    if not v_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Source video '{v_path}' does not exist on disk. Snapshot is preserved."
        )

    out_name = f"clip_{result_id}.mp4"
    clip_file = CLIPS_DIR / out_name

    # Check if already cut
    if not clip_file.exists() or clip_file.stat().st_size == 0:
        extracted = extract_video_clip(
            video_path=v_path,
            output_path=clip_file,
            start_s=offset_s,
            end_s=offset_s + 3.0,
            padding_s=2.0
        )
        if not extracted or not Path(extracted).exists():
            raise HTTPException(status_code=500, detail="Failed to extract video clip via ffmpeg.")

    return FileResponse(clip_file, media_type="video/mp4")


@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    camera: str = Form(...),
    start_time: str = Form(...),
    rotation: int = Form(0)
):
    """
    Accept video file upload + camera name + start time and save to footage/uploads.
    """
    clean_filename = Path(file.filename).name
    save_path = UPLOADS_DIR / clean_filename

    contents = await file.read()
    with open(save_path, "wb") as f:
        f.write(contents)

    # Verify duration & resolution with ffprobe if available
    fps = 30.0
    width = 1920
    height = 1080
    duration_s = 60.0

    # Insert into videos table
    engine = get_search_engine()
    now_iso = datetime.datetime.now().isoformat()
    rel_path = f"footage/uploads/{clean_filename}".replace("\\", "/")

    with engine.conn:
        engine.conn.execute(
            """
            INSERT INTO videos (path, camera, start_time, fps, width, height, duration_s, start_source, rotation, indexed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'upload', ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                camera=excluded.camera,
                start_time=excluded.start_time,
                rotation=excluded.rotation,
                indexed_at=excluded.indexed_at;
            """,
            (rel_path, camera, start_time, fps, width, height, duration_s, rotation, now_iso)
        )

    # Refresh known cameras in parser
    engine.reload_parser()

    return {
        "status": "uploaded",
        "filename": clean_filename,
        "path": rel_path,
        "camera": camera,
        "start_time": start_time,
        "size_bytes": len(contents)
    }


@app.get("/health")
def health():
    """
    Return model status, GPU memory usage, and index statistics.
    """
    total_mib, used_mib, free_mib = get_nvml_vram_info()
    engine = get_search_engine()

    cur = engine.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM videos;")
    n_videos = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM tracks;")
    n_tracks = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM frames;")
    n_frames = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM aliases;")
    n_aliases = cur.fetchone()[0]

    return {
        "status": "healthy",
        "models": {
            "detector": "yolov8s-worldv2",
            "embedder": "google/siglip-base-patch16-224",
            "device": engine.embedder.device,
            "dtype": str(engine.embedder.dtype)
        },
        "vram": {
            "total_mib": total_mib,
            "used_mib": used_mib,
            "free_mib": free_mib
        },
        "index_stats": {
            "videos_count": n_videos,
            "tracks_count": n_tracks,
            "frames_count": n_frames,
            "aliases_count": n_aliases
        }
    }


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
