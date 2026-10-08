# MULTIStream: Multi-Stream Video Intelligence with Conversational Query

Multi-camera CCTV intelligence system that indexes video feeds using open-vocabulary detection (YOLO-World), multi-object tracking (ByteTrack), and vision-language embeddings (SigLIP), enabling natural-language spatial and temporal search with verifiable visual evidence (annotated snapshots and on-demand video clips).

Built for HackNex 2026 (Problem HNX26EPS05).

---

## 1. System Features & Architecture

- **Open-Vocabulary Tracking**: YOLO-World detection + ByteTrack across arbitrary camera feeds and rotations.
- **Vision-Language Search**: Google SigLIP (`google/siglip-base-patch16-224`) embeddings for sub-350ms cosine similarity search.
- **Spatial ROI & Temporal Parsing**: Rule-based and LLM query parsing with time window resolution and camera polygon spatial filtering.
- **Clarify-Once Memory**: Disambiguates unknown locations with user confirmation and stores aliases permanently in SQLite WAL database.
- **On-Demand Clip Extraction**: Slices relevant video segments with boundary padding without re-encoding the entire feed.
- **Live Ingestion & Stream Recording**: RTSP and simulated video stream recording (`scripts/record.py`) with continuous folder monitoring (`scripts/watch.py`).
- **Cross-Camera Re-Identification (ReID)**: Automatically links tracks across cameras to construct spatio-temporal trajectories.
- **Standing Queries & Alert Triggers**: Persistent monitoring rules with automated event triggering and snapshots.
- **Web Interface & REST API**: High-performance FastAPI server and interactive browser UI for search, upload, and snippet playback.
- **100% Offline Capability**: Fully air-gapped capable (`PARSER_PROVIDER=rules`, `RERANK_PROVIDER=off`), 0 bytes transmitted externally.

---

## 2. Prerequisites & Environment

- **OS**: Windows / Linux
- **GPU**: NVIDIA GPU (CUDA-compatible, e.g. RTX 3050 4 GB VRAM)
- **Software**: Python 3.11+, `ffmpeg` & `ffprobe`
- **Key Python Packages**: `torch>=2.6`, `transformers>=4.40`, `ultralytics>=8.3`, `fastapi>=0.110`, `opencv-python`

---

## 3. Quick Start & Demo

Run the automated end-to-end demonstration script:

```powershell
.venv\Scripts\python.exe scripts/demo.py
```

Launch the web application:
```powershell
.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```
Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

---

## 4. CLI Commands & Workflows

### Ingestion
```powershell
# Ingest single video or folder
.venv\Scripts\python.exe scripts/ingest.py --video footage/test_video01.mp4 --index-dir index_base

# Folder watching & live incremental ingest
.venv\Scripts\python.exe scripts/watch.py --watch-dirs footage/recorded --once
```

### Natural Language Search
```powershell
# Search with clip cutting
.venv\Scripts\python.exe scripts/query.py --query "pedestrian on cam_landscape" --cut-clips

# Search with temporal constraint
.venv\Scripts\python.exe scripts/query.py --query "car yesterday" --top-k 5
```

### Camera Aliases & Spatial Polygon
```powershell
# Define camera alias with spatial ROI
.venv\Scripts\python.exe scripts/alias.py --add "driveway" --camera cam_landscape --polygon "[[0.1, 0.2], [0.8, 0.2], [0.8, 0.9], [0.1, 0.9]]"

# List saved aliases
.venv\Scripts\python.exe scripts/alias.py --list
```

### Cross-Camera Re-Identification
```powershell
# Re-identify track across cameras
.venv\Scripts\python.exe scripts/reid.py --track-id test_landscape_trk_11 --min-sim 0.70

# Discover all candidate camera hops
.venv\Scripts\python.exe scripts/reid.py --min-sim 0.65 --top-k 5
```

### Standing Queries & Alerts
```powershell
# Add standing alert rule
.venv\Scripts\python.exe scripts/alerts.py --add --name "Vehicle Movement" --query "truck or car or suv" --min-score 0.04

# Evaluate rules against tracks
.venv\Scripts\python.exe scripts/alerts.py --evaluate

# List triggered events
.venv\Scripts\python.exe scripts/alerts.py --list-events
```

---

## 5. Testing & Evaluation

```powershell
# Run unit and integration tests
.venv\Scripts\python.exe -m pytest tests/ -v

# Run evaluation benchmark
.venv\Scripts\python.exe scripts/eval.py
```

---

## 6. Documentation & Architecture
- [DEMO.md](DEMO.md) - Complete end-to-end user walkthrough.
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture and component interactions.
- [DECISIONS.md](DECISIONS.md) - Architecture decision records (ADRs).
- [RULES.md](RULES.md) - Verification and honesty protocols.
