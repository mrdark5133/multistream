# Project Tasks & Verification Checklist

Rules: Check a box ONLY when the acceptance test for that task has passed. Include a link/reference to the corresponding report section.

---

## Phase 0: Environment, Docs, Model Feasibility
- [x] 0.1 Machine inspection (OS, Python, `nvidia-smi`, disk space, ffmpeg, git) - [PHASE_0_REPORT.md Section 3](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#3-environment-actually-used)
- [x] 0.2 Virtual environment & dependencies installed (`requirements.txt`, PyTorch CUDA verified) - [PHASE_0_REPORT.md Section 4](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#4-commands-run-and-raw-output)
- [x] 0.3 Core documentation created (`PLAN.md`, `TASKS.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `RULES.md`, `EVAL.md`, `README.md`, `.gitignore`, `.env.example`, `REPORTS/`) - [PHASE_0_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#2-what-was-done)
- [x] 0.4 Download and load YOLO-World (`yolov8s-worldv2.pt`) and apply `vocab.yaml` (~60 classes) - [PHASE_0_REPORT.md Section 5](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#5-measurements)
- [x] 0.5 Co-load `google/siglip-so400m-patch14-384` (FP16) with YOLO-World on GPU; measure peak VRAM - [PHASE_0_REPORT.md Section 5](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#5-measurements)
- [x] 0.6 Evaluate fallback options if SO400M fails or exceeds 4 GB VRAM; record choice in `DECISIONS.md` - [PHASE_0_REPORT.md Section 5](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#5-measurements)
- [x] 0.7 Validate image-text semantic similarity ranking on real sample images - [PHASE_0_REPORT.md Section 6](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#6-acceptance-tests)
- [x] 0.8 Verify LLM providers (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`) and record response status & latency - [PHASE_0_REPORT.md Section 6](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#6-acceptance-tests)
- [x] 0.9 Write and execute `scripts/check_env.py` - [PHASE_0_REPORT.md Section 4](file:///c:/projects/MULTIStream/REPORTS/PHASE_0_REPORT.md#4-commands-run-and-raw-output)
- [x] Phase 0 Report generated and approved (`REPORTS/PHASE_0_REPORT.md`)

---

## Phase 1: Ingest Pipeline
- [x] 1.1 Implement SQLite schema and `config` table in WAL mode - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.2 Implement video start time resolution fallback chain with unit tests - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.3 Detection + ByteTrack tracking with configurable stride (default: 5) - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.4 Best-frame selection per track and annotated snapshot creation - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.5 Crop embedding extraction with automatic OOM batch backoff - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.6 Whole-frame embedding extraction (~1 frame every 2s) into `frames` table - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.7 Incremental indexing with duplicate prevention - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.8 CLI tool `scripts/ingest.py` logging duration, tracks, frames, wall time, VRAM - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.9 Multi-video tracker isolation test preventing track state leakage - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.10 Video rotation normalization: read rotation metadata per video (tags/side data), apply rotation before detection/embedding, and store applied rotation in `videos` record - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.11 Benchmark image tower only (3 warm-up + 20 timed runs, batch sizes 1/8/16, both embedders: SigLIP-Base and SigLIP-SO400M) on crops extracted from upright clips (`test_video01-03.mp4`) - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.12 NVML free-VRAM guard check before each model load, and execute YOLO `set_classes` on CPU or explicitly delete/free CLIP text encoder (+443 MiB) afterward - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] 1.13 High-resolution landscape CCTV input handling (evaluate imgsz=1280 vs default imgsz=640 for small-object recall on 2.5K/4K CCTV frames) - [PHASE_1_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_1_REPORT.md)
- [x] Phase 1 Report generated and approved (`REPORTS/PHASE_1_REPORT.md`)

---

## Phase 2: Query Engine (Search, No UI)
- [x] 2.1 Implement `parse_query` with pluggable providers (`claude`, `gemini`, `rules`) and context bundle - [PHASE_2_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_2_REPORT.md)
- [x] 2.2 Implement relative time parsing against DB "now" timestamp with unit tests - [PHASE_2_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_2_REPORT.md)
- [x] 2.3 Implement SigLIP cosine vector search with prompt tuning and spatio-temporal filtering - [PHASE_2_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_2_REPORT.md)
- [x] 2.4 Construct structured result objects with snapshot paths and clip descriptors - [PHASE_2_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_2_REPORT.md)
- [x] 2.5 Implement `ffmpeg` on-demand clip extraction with boundary padding - [PHASE_2_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_2_REPORT.md)
- [x] 2.6 CLI tool `scripts/query.py` returning top ranked results - [PHASE_2_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_2_REPORT.md)
- [x] 2.7 Measure stage latency profiling (parse, embed, search, clip cut) - [PHASE_2_REPORT.md Section 2](file:///c:/projects/MULTIStream/REPORTS/PHASE_2_REPORT.md)
- [x] Phase 2 Report generated and approved (`REPORTS/PHASE_2_REPORT.md`)

---

## Phase 3: Clarify-Once Memory
- [x] 3.1 Implement `aliases` table with normalized name matching and polygon storage
- [x] 3.2 Implement clarify flow: return clarification request for unknown referents and store `save_alias`
- [x] 3.3 Implement normalized polygon bounding-box center spatial filter
- [x] 3.4 Automated test suite:
  - [x] 3.4.a Unknown referent triggers clarification exactly once
  - [x] 3.4.b Stored alias never asks again across paraphrased queries
  - [x] 3.4.c Cross-process server restart persistence test
  - [x] 3.4.d Known camera names bypass clarification
  - [x] 3.4.e Polygon filter correctness on track centers
- [x] 3.5 Disallow LLM/context guessing for unknown locations
- [x] Phase 3 Report generated and approved (`REPORTS/PHASE_3_REPORT.md`)

---

## Phase 4: API and Chat UI
- [x] 4.1 Implement FastAPI endpoints (`/ask`, `/alias`, `/cameras`, `/snapshot/{id}`, `/clip/{id}`, `/upload`, `/health`) with in-memory multi-turn session history
- [x] 4.2 Minimal static chat UI with result cards, video players, and polygon annotation canvas
- [x] 4.3 Automated terminal-based API testing (`curl` / `httpx`) without browser automation
- [x] 4.4 Formulate MANUAL CHECK protocol for human browser inspection
- [x] Phase 4 Report generated and approved (`REPORTS/PHASE_4_REPORT.md`)

---

## Phase 5: Evaluation and Ablation (Research Contribution)
- [x] 5.1 Implement evaluation harness `scripts/eval.py` and document metrics in `EVAL.md`
- [x] 5.2 Split test queries into dev and held-out evaluation sets
- [x] 5.3 Implement fair whole-frame retrieval baseline
- [x] 5.4 Execute full ablation matrix across all pipeline components
- [x] 5.5 Profile latency and index storage footprint per ablation row
- [x] 5.6 Output structured evaluation results in `eval/results/` and format report tables
- [x] Phase 5 Report generated and approved (`REPORTS/PHASE_5_REPORT.md`)

---

## Phase 6: Accuracy Improvements (Evidence-Driven)
- [ ] 6.1 Systematic failure case error analysis
- [ ] 6.2 Evaluate `yolov8m-worldv2.pt` if detection recall is bottlenecked
- [ ] 6.3 Optional VLM reranking on top 10 candidates with measured latency & cloud egress disclosure
- [ ] 6.4 Query prompt adjustments or attribute verification
- [ ] Phase 6 Report generated and approved (`REPORTS/PHASE_6_REPORT.md`)

---

## Phase 7: Stretch Goals and Hardening
- [ ] 7.1 Unified `watch.py` folder watcher and RTSP `record.py`
- [ ] 7.2 Inter-camera track re-identification and spatio-temporal timeline generation
- [ ] 7.3 Standing query and event alert triggers
- [ ] 7.4 Privacy audit & offline performance verification (`PARSER_PROVIDER=rules`, `RERANK_PROVIDER=off`)
- [ ] 7.5 Project polish, documentation cleanup, and demo script
- [ ] Phase 7 Report generated and approved (`REPORTS/PHASE_7_REPORT.md`)
