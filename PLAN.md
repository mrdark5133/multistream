# Project Plan: HNX26EPS05 Multi-Stream Video Intelligence

## 1. Goal & Mission
Build a production-grade multi-stream video intelligence system with conversational query capabilities for HackNex 2026 (Problem HNX26EPS05).
The system ingests recorded (and live RTSP) video from multiple cameras, detects and tracks objects, extracts best-frame visual representations, indexes embeddings and metadata, and responds to natural-language queries (e.g., "did a red car pass through the main gate in the last hour?").
Every query response strictly resolves to:
- Specific camera name
- Timestamp (absolute and offset)
- Visual evidence: annotated snapshot image and clipped video segment
Answers without traceable sources are treated as failures.

## 2. System Architecture & Two-Phase Design
### 2.1 Ingestion Phase (Heavy, Offline/Background per video)
1. Video file intake (`footage/` or `live_footage/` via RTSP recorder).
2. Frame sampling with configurable stride (default: 5 frames).
3. Open-vocabulary object detection using YOLO-World (`yolov8s-worldv2.pt`) loaded with ~60 CCTV classes (`config/vocab.yaml`).
4. Multi-object tracking with ByteTrack (strictly isolated per video).
5. Track aggregation: select best frame per track (max area × confidence), produce annotated snapshot (JPEG, ~640px wide).
6. Feature embedding: SigLIP embedder on best object crops (batched with automatic OOM fallback).
7. Whole-frame fallback: embed full frames at ~1 frame per 2 seconds into `frames` table.
8. Persistence: write structured metadata and embeddings to SQLite index.

### 2.2 Query Phase (Fast, Synchronous)
1. Query input received via CLI or FastAPI.
2. Query parsing into structured JSON (`object_description`, `location_referent`, `time_range`, `camera`) via provider interface (`PARSER_PROVIDER`: `claude`, `gemini`, or `rules` fallback) with context bundle (cameras, known aliases, relative time reference).
3. Clarify-once Referent Resolution:
   - If referent matches known camera or existing alias in `aliases` table, resolve immediately.
   - If unknown, halt and return clarification request to user. Upon resolution (camera selection + optional polygon), store permanently in SQLite.
4. Temporal parsing: compute start/end absolute timestamps relative to latest indexed timestamp in DB.
5. Vector retrieval: SigLIP text embedding compared against track crop embeddings and whole-frame embeddings.
6. Filtering & Spatial Verification: filter by camera, time range, and polygon inclusion (bbox center).
7. Merging & Deduplication: merge near-duplicate detections of the same object across temporal proximity.
8. Evidence generation: generate snapshot link and extract video clip on demand using `ffmpeg`.

## 3. Models & Hardware Budget
- **Target Hardware**: NVIDIA RTX 3050 (4 GB VRAM), 24 GB System RAM.
- **Detector**: YOLO-World (`yolov8s-worldv2.pt`), expandable to `yolov8m-worldv2.pt` if recall requires.
- **Tracker**: ByteTrack (Ultralytics).
- **Vision-Language Embedder**: `google/siglip-so400m-patch14-384` (FP16 on GPU if it fits in 4 GB VRAM; otherwise image-tower only or fallback to `google/siglip-base-patch16-224`).
- **Query Parser**: Pluggable interface (`claude`, `gemini`, `rules`). Rule-based parser acts as 100% offline, zero-cost fallback.
- **Reranker (Phase 6 Optional)**: Vision-language judge (`gemini`, `claude`, or `off`) on top 10 snapshots.

## 4. SQLite Schema (WAL Mode)
- `config(key, value)`
- `videos(path PRIMARY KEY, camera, start_time, fps, width, height, duration_s, start_source, indexed_at)`
- `tracks(id, video, camera, track_id, label, t_start, t_end, t_best, offset_start, offset_end, bbox_px, bbox_norm, snapshot, emb)`
- `frames(id, video, camera, t_abs, offset_s, snapshot, emb)`
- `aliases(name, camera, polygon_norm NULL, created_at)`
- Indexes on `(camera, t_start, t_end)`.

## 5. Execution Phases
- **Phase 0**: Environment setup, documentation, hardware verification, model feasibility & VRAM profiling.
- **Phase 1**: Ingest pipeline (YOLO-World, ByteTrack, best-frame snapshot, crop & whole-frame SigLIP embedding, SQLite indexing, tracker isolation).
- **Phase 2**: Query engine (query parser with context bundle, relative time parsing, SigLIP text search, duplicate merging, ffmpeg clip extraction, latency profiling).
- **Phase 3**: Clarify-once memory (alias normalization, polygon spatial filtering, cross-process and restart persistence).
- **Phase 4**: API and chat UI (FastAPI endpoints, session conversation history, static browser UI with polygon canvas, curl/httpx verification).
- **Phase 5**: Evaluation and ablation (evaluation harness `scripts/eval.py`, baseline frame retrieval vs full pipeline, ablation rows on dev and held-out splits).
- **Phase 6**: Accuracy improvements (failure analysis, detector upgrade or VLM reranker if evidence warrants).
- **Phase 7**: Stretch goals & hardening (unified watcher, RTSP recorder, cross-camera re-ID, local privacy audit, demo script).

## 6. Key Risks and Mitigations
1. **4 GB VRAM limit with SigLIP SO400M + YOLO-World**:
   - *Mitigation*: Rigorously profile peak VRAM in Phase 0. If SO400M exceeds budget or leaves insufficient headroom for batching, switch to `siglip-base-patch16-224` (768-dim) or execute text tower on CPU and image tower in FP16. Document all measurements in `DECISIONS.md`.
2. **LLM Parser Rate Limits or Missing API Keys**:
   - *Mitigation*: Rule-based regex/heuristic parser must be fully implemented and reliable as the default fallback.
3. **Tracking ID Bleed Between Videos**:
   - *Mitigation*: Re-initialize tracker state explicitly per video; verify via multi-video isolation unit tests.
4. **Referent Hallucination**:
   - *Mitigation*: Strictly forbid LLM or context heuristics from guessing unknown locations. If not in DB `aliases` or camera list, enforce clarification.
