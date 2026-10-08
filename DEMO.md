# MULTIStream End-to-End System Demo Walkthrough

This document guides you through running the complete, end-to-end multi-camera video intelligence workflow of **MULTIStream**.

---

## 1. Quick Automated Demo Run

Run the automated Python demonstration script to see all components in action:

```powershell
.venv\Scripts\python.exe scripts/demo.py
```

This runs:
1. **Catalog & Database Status**: Summarizes indexed videos, timestamps, tracks, and sampled frames.
2. **Vector Similarity Search**: Parses natural language query `"pedestrian on cam_landscape"`, applies camera filters, and computes cosine similarities in under 350 ms.
3. **Cross-Camera Re-identification (ReID)**: Automatically links tracks across different cameras over time to trace physical entities.
4. **Standing Queries & Alerts**: Evaluates alert rules against indexed tracks and lists event triggers.
5. **Interactive Server Status**: Confirms live status of the FastAPI backend and browser interface.

---

## 2. Interactive Web Application

Launch or access the running server:

```powershell
.venv\Scripts\python.exe -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Open your browser to:
- **Web UI**: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- **API Docs (Swagger)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Web Features:
- **Search Bar**: Query with natural language queries like `"blue car yesterday"`, `"person at cam_landscape"`, or `"truck this morning"`.
- **Clarify-Once Alias Modal**: Try querying `"red car at driveway"` when "driveway" is unmapped; an interactive modal lets you map "driveway" to a camera and draw a spatial polygon.
- **Video Upload**: Ingest new footage directly via drag-and-drop with instant tracking and indexing.
- **Video Snippet Player**: Click on any search card to stream the exact on-demand extracted MP4 snippet.

---

## 3. CLI Search & Snippet Extraction

Query the multi-camera database directly from PowerShell:

```powershell
# Basic query with automated clip cutting
.venv\Scripts\python.exe scripts/query.py --query "pedestrian on cam_landscape" --cut-clips

# Query with temporal expression
.venv\Scripts\python.exe scripts/query.py --query "car yesterday" --top-k 5

# Search with custom spatial polygon filter
.venv\Scripts\python.exe scripts/query.py --query "red car" --polygon "[0.1, 0.1, 0.8, 0.8]"
```

---

## 4. Live Stream Recording & Unified Folder Watching

### Step A: Record simulated or live RTSP streams into chunks
```powershell
.venv\Scripts\python.exe scripts/record.py --stream footage/test_video01.mp4 --camera cam_sim1 --output-dir footage/recorded --segment-duration 5 --max-duration 15
```

### Step B: Watch folder and automatically index new video segments
```powershell
# Continuous monitoring
.venv\Scripts\python.exe scripts/watch.py --watch-dirs footage/recorded --poll-interval 2.0

# Or single-pass batch scan
.venv\Scripts\python.exe scripts/watch.py --watch-dirs footage/recorded --once
```

---

## 5. Cross-Camera Track Re-identification (ReID)

Trace objects as they move between cameras:

```powershell
# Re-identify a specific track across other cameras
.venv\Scripts\python.exe scripts/reid.py --track-id test_landscape_trk_11 --min-sim 0.70

# Discover all candidate cross-camera associations across all feeds
.venv\Scripts\python.exe scripts/reid.py --min-sim 0.65 --top-k 5
```

---

## 6. Standing Queries & Event Alert Triggers

Register persistent rules that monitor incoming footage:

```powershell
# 1. Register a standing alert rule
.venv\Scripts\python.exe scripts/alerts.py --add --name "Vehicle Movement" --query "truck or car or suv" --min-score 0.04

# 2. List registered rules
.venv\Scripts\python.exe scripts/alerts.py --list-rules

# 3. Evaluate rules against newly ingested tracks
.venv\Scripts\python.exe scripts/alerts.py --evaluate

# 4. Review triggered events
.venv\Scripts\python.exe scripts/alerts.py --list-events
```

---

## 7. 100% Offline & Air-Gapped Privacy Verification

MULTIStream operates completely locally without sending any data off-device:
- **Query Parser**: Set `PARSER_PROVIDER=rules` for zero cloud dependency.
- **Vision Embeddings**: SigLIP runs on local GPU/CPU (`cuda:0`).
- **Object Detection & Tracking**: YOLO-World + ByteTrack runs locally.
- **Database**: SQLite WAL mode file `index_base/index.db`.
- **Reranker**: Set `RERANK_PROVIDER=off` to ensure 0 external API calls.
