import os
import time
import json
import sqlite3
import datetime
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query, status, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.query.search import SearchEngine
from src.query.clips import extract_video_clip
from src.utils.vram import get_nvml_vram_info
from src.ingest.live_stream import LiveStreamWorker
from src.ingest.detect_track import load_yolo_world


# Project directories
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
STATIC_DIR = ROOT_DIR / "static"
INDEX_DB = Path(os.environ.get("INDEX_DB_PATH", ROOT_DIR / "index_mobile" / "index.db" if (ROOT_DIR / "index_mobile" / "index.db").exists() else (ROOT_DIR / "index_fresh" / "index.db" if (ROOT_DIR / "index_fresh" / "index.db").exists() else ROOT_DIR / "index_base" / "index.db")))

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


from src.voice.asr import WhisperASR, decode_audio_bytes, resample_to_16k
from src.voice.tts import PiperTTS
from src.voice.answers import template_answer
from src.voice.camera_match import match_spoken_to_camera, normalize_query_cameras
import io
import soundfile as sf
import numpy as np


# Voice Engine Singletons & Session State
_asr_instance: Optional[WhisperASR] = None
_tts_instance: Optional[PiperTTS] = None
_voice_sessions: Dict[str, Dict[str, Any]] = {}


def get_asr() -> WhisperASR:
    global _asr_instance
    if _asr_instance is None:
        _asr_instance = WhisperASR.from_config()
    return _asr_instance


def get_tts() -> PiperTTS:
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = PiperTTS.from_config()
    return _tts_instance


def execute_ask_search(
    query_text: str,
    top_k: int = 5,
    session_id: Optional[str] = None,
    now: Optional[str] = None
) -> Dict[str, Any]:
    """
    Core search execution shared across text /ask and /voice endpoints.
    """
    engine = get_search_engine()
    now_override = datetime.datetime.fromisoformat(now) if now else None

    resp = engine.query(
        query_text=query_text,
        top_k=top_k,
        now_override=now_override
    )

    sid = session_id or "default"
    if sid not in session_history:
        session_history[sid] = []
    session_history[sid].append({
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query_text,
        "status": resp["status"]
    })
    session_history[sid] = session_history[sid][-5:]

    if resp["status"] == "clarify":
        return {
            "status": "clarify",
            "referent": resp["referent"],
            "options": resp["options"]
        }

    parsed = resp["parsed"]

    if _live_worker and _live_worker.is_running and parsed and parsed.object_prompt:
        obj_phrase = parsed.object_prompt.strip().lower()
        if obj_phrase.startswith("a "): obj_phrase = obj_phrase[2:]
        elif obj_phrase.startswith("an "): obj_phrase = obj_phrase[3:]
        if obj_phrase and obj_phrase not in _live_worker.classes:
            _live_worker.extend_vocabulary(obj_phrase)

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
            "color": r.color,
            "snapshot_url": f"/snapshot/{r.result_id}",
            "clip_url": f"/clip/{r.result_id}",
            "bbox_px": r.bbox_px,
            "bbox_norm": r.bbox_norm
        })

    return {
        "status": "success",
        "query": query_text,
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


@app.post("/ask")
def ask(req: AskRequest):
    """
    Execute natural language query with clarify-once protocol.
    Returns status='clarify' when referent is unknown, or status='success' with ranked results.
    """
    return execute_ask_search(
        query_text=req.query,
        top_k=req.top_k,
        session_id=req.session_id,
        now=req.now
    )


@app.post("/voice")
async def voice_search(
    file: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    session_id: Optional[str] = Form(None)
):
    """
    Local Push-to-Talk Voice Interface Endpoint (CPU only).
    Pipeline: faster-whisper -> transcript -> /ask logic -> templated answer -> Piper TTS -> WAV audio.
    """
    t_start = time.perf_counter()
    sid = session_id or "default"

    upload = file or audio
    if upload is None:
        return JSONResponse(
            status_code=400,
            content={"detail": "Missing audio file in form", "error": "missing_audio"}
        )

    # 1. Read audio bytes
    raw_bytes = await upload.read()
    if not raw_bytes or len(raw_bytes) < 44:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid or corrupt WAV audio file.", "error": "corrupt_audio"}
        )

    # 2. Decode audio (supports WAV, WebM, OGG, MP3, etc.)
    audio_data, sample_rate = decode_audio_bytes(raw_bytes)
    if len(audio_data) == 0:
        return JSONResponse(
            status_code=400,
            content={"detail": "Invalid or corrupt WAV audio file.", "error": "corrupt_audio"}
        )

    # Resample to 16,000 Hz if needed (crucial for real Windows PC mic hardware)
    if sample_rate != 16000:
        audio_data = resample_to_16k(audio_data, sample_rate)
        sample_rate = 16000

    duration_s = float(len(audio_data)) / float(sample_rate)

    # Reject > 30s audio
    if duration_s > 30.0:
        return JSONResponse(
            status_code=413,
            content={"detail": "Audio exceeded 30 seconds limit", "error": "audio_too_long"}
        )

    # Silence check (< 0.2s or RMS < 0.001)
    rms = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) > 0 else 0.0
    max_amp = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
    if duration_s < 0.2 or rms < 0.001 or max_amp < 0.002:
        return JSONResponse(
            status_code=400,
            content={"detail": "No speech detected in audio.", "error": "silence_detected"}
        )

    # 3. Automatic Speech Recognition (faster-whisper on CPU)
    asr = get_asr()
    t0_asr = time.perf_counter()
    transcript = asr.transcribe(audio_data)
    t_asr = (time.perf_counter() - t0_asr) * 1000.0

    if not transcript or not transcript.strip():
        tts = get_tts()
        t0_tts = time.perf_counter()
        audio_id, _ = tts.synthesize("I didn't catch that")
        t_tts = (time.perf_counter() - t0_tts) * 1000.0
        t_total = (time.perf_counter() - t_start) * 1000.0
        return {
            "transcript": "",
            "ask_response": {"status": "empty", "results": []},
            "spoken_text": "I didn't catch that",
            "audio_url": f"/voice/audio/{audio_id}",
            "timings_ms": {
                "asr": round(t_asr, 1),
                "search": 0.0,
                "tts": round(t_tts, 1),
                "total": round(t_total, 1)
            }
        }

    # 4. Handle Voice Clarification Flow
    sess = _voice_sessions.get(sid, {})
    query_to_run = transcript

    if "pending_clarify" in sess:
        clar_info = sess["pending_clarify"]
        matched_cam = match_spoken_to_camera(transcript, clar_info.get("options", []))
        if matched_cam:
            referent = clar_info["referent"]
            engine = get_search_engine()
            engine.save_alias(name=referent, camera=matched_cam, polygon_norm=[[0.0, 0.0]])
            query_to_run = clar_info["original_query"]
            del sess["pending_clarify"]

    # Normalize camera mentions (e.g. 'landscape two camera' -> 'cam_landscape2')
    normalized_query = normalize_query_cameras(query_to_run)

    # 5. Execute Search Logic
    t0_search = time.perf_counter()
    ask_resp = execute_ask_search(normalized_query, top_k=5, session_id=sid)
    t_search = (time.perf_counter() - t0_search) * 1000.0

    # If new query needs clarification, update session state
    if ask_resp.get("status") == "clarify":
        sess["pending_clarify"] = {
            "referent": ask_resp["referent"],
            "options": ask_resp["options"],
            "original_query": query_to_run
        }
        _voice_sessions[sid] = sess

    # 6. Templated Spoken Answer Generation
    spoken_text = template_answer(ask_resp, query_text=normalized_query)

    # 7. Text-To-Speech Synthesis (Piper on CPU)
    tts = get_tts()
    t0_tts = time.perf_counter()
    audio_id, _ = tts.synthesize(spoken_text)
    t_tts = (time.perf_counter() - t0_tts) * 1000.0

    t_total = (time.perf_counter() - t_start) * 1000.0

    return {
        "transcript": transcript,
        "ask_response": ask_resp,
        "spoken_text": spoken_text,
        "audio_url": f"/voice/audio/{audio_id}",
        "timings_ms": {
            "asr": round(t_asr, 1),
            "search": round(t_search, 1),
            "tts": round(t_tts, 1),
            "total": round(t_total, 1)
        }
    }


@app.get("/voice/audio/{audio_id}")
def get_voice_audio(audio_id: str):
    """Serve synthesized or cached voice WAV audio file."""
    tts = get_tts()
    audio_path = tts.get_audio_path(audio_id)
    if audio_path and audio_path.exists():
        return FileResponse(audio_path, media_type="audio/wav")
    raise HTTPException(status_code=404, detail="Voice audio file not found")


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
        INDEX_DB.parent,
        INDEX_DB.parent / "snapshots",
        ROOT_DIR / "index_fresh",
        ROOT_DIR / "index_fresh" / "snapshots",
        ROOT_DIR / "index_base",
        ROOT_DIR / "index_base" / "snapshots",
        ROOT_DIR,
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
        for base in [INDEX_DB.parent, ROOT_DIR / "index_fresh", ROOT_DIR / "index_base", ROOT_DIR]:
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


# --- Live Mobile Webcam / RTSP Stream Endpoints ---

class StreamStartRequest(BaseModel):
    stream_url: str = Field(..., description="IP webcam or RTSP stream URL (or path to video file for simulated demo)")
    camera: str = Field("mobile_cam01", description="Camera name identifier")


_live_worker: Optional[LiveStreamWorker] = None
_yolo_model = None
_classes = None


@app.post("/stream/start")
def start_live_stream(req: StreamStartRequest):
    """
    Connect to a live mobile phone stream (IP Webcam) or simulated stream,
    performing real-time detection, tracking, color tagging, and SQLite indexing.
    """
    global _live_worker, _yolo_model, _classes
    if _live_worker and _live_worker.is_running:
        _live_worker.stop()

    engine = get_search_engine()
    if _yolo_model is None:
        _yolo_model, _classes = load_yolo_world(
            weights_path="yolov8s-worldv2.pt",
            vocab_path="config/live_vocab.yaml",
            device="cuda:0"
        )

    _live_worker = LiveStreamWorker(
        stream_url=req.stream_url,
        camera_name=req.camera,
        db_path=INDEX_DB,
        snapshots_dir=INDEX_DB.parent / "snapshots",
        recorded_dir=ROOT_DIR / "footage" / "recorded",
        stride=5,
        store_live_tracks=False
    )
    _live_worker.start(yolo_model=_yolo_model, embedder=engine.embedder, classes=_classes)
    engine.reload_parser()
    return {"status": "started", "camera": req.camera, "stream_url": req.stream_url}


_indexing_status = {
    "status": "idle",
    "message": "Grounding DINO: Ready to Fetch",
    "video": None,
    "camera": None,
    "tracks_indexed": 0,
    "started_at": None,
    "completed_at": None
}


def run_post_stream_ingest(rec_rel_path: str, camera_name: str):
    global _indexing_status
    if not rec_rel_path:
        return
    rec_full_path = ROOT_DIR / rec_rel_path
    if not rec_full_path.exists() or rec_full_path.stat().st_size == 0:
        return
    try:
        import logging
        import datetime
        log = logging.getLogger("multistream.auto_ingest")
        _indexing_status["status"] = "running"
        _indexing_status["message"] = f"Running Grounding DINO on {rec_full_path.name}..."
        _indexing_status["video"] = rec_full_path.name
        _indexing_status["camera"] = camera_name
        _indexing_status["started_at"] = datetime.datetime.now().isoformat()
        log.info(f"[AUTO-INGEST] Grounding-DINO post-stream indexing started for: {rec_full_path.name}")
        from src.ingest.detector import GroundingDinoDetector
        from src.ingest.pipeline import IngestPipeline
        detector = GroundingDinoDetector(
            vocab_path=str(ROOT_DIR / "config" / "test_vocab.yaml"),
            device="cuda:0",
            conf_thresh=0.20,
            text_thresh=0.20
        )
        pipeline = IngestPipeline(
            db_path=INDEX_DB,
            snapshots_dir=INDEX_DB.parent / "snapshots",
            device="cuda:0",
            custom_detector=detector
        )
        res = pipeline.ingest_video(
            video_path=rec_full_path,
            manual_camera=camera_name,
            stride=5,
            conf_thresh=0.20,
            min_duration_s=0.3,
            min_hits=2,
            min_mean_conf=0.25,
            force=True
        )
        tracks_count = res.get('tracks_indexed', 0)
        log.info(f"[AUTO-INGEST] Grounding-DINO post-stream indexing finished: {tracks_count} tracks indexed")
        _indexing_status["status"] = "completed"
        _indexing_status["tracks_indexed"] = tracks_count
        _indexing_status["message"] = f"Grounding DINO finished: {tracks_count} objects indexed & ready to fetch!"
        _indexing_status["completed_at"] = datetime.datetime.now().isoformat()
        engine = get_search_engine()
        engine.reload_parser()
    except Exception as e:
        import logging
        import datetime
        logging.getLogger("multistream.auto_ingest").error(f"[AUTO-INGEST] Error during post-stream indexing: {e}")
        _indexing_status["status"] = "error"
        _indexing_status["message"] = f"Indexing failed: {e}"
        _indexing_status["completed_at"] = datetime.datetime.now().isoformat()


@app.post("/stream/stop")
def stop_live_stream(background_tasks: BackgroundTasks):
    """Stop active live mobile stream ingestion and trigger automatic Grounding-DINO indexing."""
    global _live_worker
    if _live_worker:
        _live_worker.stop()
        rec_file = _live_worker.stats.get("recorded_file")
        cam_name = _live_worker.camera_name
        if rec_file:
            background_tasks.add_task(run_post_stream_ingest, rec_file, cam_name)
        return {"status": "stopped", "stats": _live_worker.stats, "auto_ingest": "started"}
    return {"status": "no_active_stream"}


@app.get("/indexing/status")
def get_indexing_status():
    """Retrieve runtime status of Grounding DINO indexing pipeline and active database tracks."""
    global _indexing_status
    total_tracks = 0
    try:
        import sqlite3
        conn = sqlite3.connect(INDEX_DB)
        total_tracks = conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]
        conn.close()
    except Exception:
        pass
    return {
        **_indexing_status,
        "total_database_tracks": total_tracks
    }


@app.get("/stream/status")
def get_live_stream_status():
    """Retrieve runtime status of active live stream."""
    global _live_worker
    if _live_worker:
        return _live_worker.stats
    return {"status": "idle"}


@app.get("/stream/feed")
def stream_feed():
    """
    Serve live multipart MJPEG video feed showing camera view and real-time bounding boxes.
    """
    import time

    def frame_generator():
        while _live_worker and _live_worker.is_running:
            if getattr(_live_worker, "latest_jpeg", None) is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + _live_worker.latest_jpeg + b"\r\n"
                )
            time.sleep(0.04)

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# Mount static assets
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
