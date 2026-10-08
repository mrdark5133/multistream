# Architectural & Implementation Decisions Log

This document records technical and design decisions, recording rationale, tradeoffs, and empirical alternatives.

---

### [2026-10-08] DECISION-001: Strict Verification & Anti-Hallucination Protocol
- **Status**: Adopted
- **Context**: 24-hour hackathon judging requires verifiable, truthful metrics. Any unmeasured claims compromise review integrity.
- **Decision**: Adopt the rules in `RULES.md` verbatim. No synthetic numbers or unverified assertions. Every report contains raw command outputs.
- **Alternatives Considered**: Fast informal prototyping. Rejected because evaluation is strictly audited by reviewer model and judges.

---

### [2026-10-08] DECISION-002: Storage Schema: Store Pointers & Snapshots, Not Video Duplicates
- **Status**: Adopted
- **Context**: Video files consume substantial disk space; copying or re-encoding raw streams on ingest causes unnecessary I/O overhead.
- **Decision**: Store `video_path` plus `offset_start` and `offset_end` in seconds. Cut MP4 clips on-demand with `ffmpeg`. Generate and store annotated JPEG snapshots at ingest time to guarantee visual evidence persists even if raw footage is moved.
- **Alternatives Considered**: Pre-cutting clips for every track during ingestion. Rejected due to extreme disk usage and ingest latency overhead.

---

### [2026-10-08] DECISION-003: Embedder Co-existence on 4 GB VRAM (RTX 3050)
- **Status**: Measured & Decided (Amended with Windows-Closed Data & Upright Footage)
- **Context**: Profiled YOLO-World (`yolov8s-worldv2.pt`) co-loaded with both SigLIP models in FP16 on upright footage (`footage/test_video01.mp4` frame).
- **Empirical Measurements**:
  1. **`google/siglip-so400m-patch14-384` (FP16)**:
     - Peak PyTorch Allocated: **2381.2 MB**
     - Condition (a) (Desktop apps open): Total Used = **3647.0 MB** / 4094.0 MB | Headroom = **246.0 MB** (6.0% buffer) -> **FAILS** the 3.4 GB rule.
     - Condition (b) (App windows closed, background processes running): Total Used = **3037.0 MB** / 4094.0 MB | Headroom = **856.0 MB** (20.9% buffer) -> **PASSES** the 3.4 GB rule.
     - Single Crop Inference: **INVALID: 464.1 ms** (single cold run on the inverted `test_gate.mp4` frame; image-tower-only benchmark with 3 warm-up + 20 timed runs across batch 1/8/16 scheduled for Phase 1 in `TASKS.md` Task 1.11).
     - WDDM Spill-to-Host Behavior: Under Windows WDDM 3.x, if allocations exceed physical VRAM, memory spills into host system RAM / pagefile, degrading inference throughput rather than immediately raising OutOfMemory. Batching crops with SO400M risks severe paging latency spikes.
  2. **`google/siglip-base-patch16-224` (FP16)**:
     - Peak PyTorch Allocated: **1086.1 MB**
     - Total System VRAM Used (Windows closed): **1670.0 MB** / 4094.0 MB | Headroom = **2223.0 MB** (54.3% buffer).
     - Total System VRAM Used (Condition a): **2279.0 MB** / 4094.0 MB | Headroom = **1614.0 MB** (39.4% buffer).
     - Single Crop Inference: **INVALID: 198.6 ms ("2.34x faster")** (single cold run on the inverted `test_gate.mp4` frame; image-tower-only benchmark with 3 warm-up + 20 timed runs across batch 1/8/16 scheduled for Phase 1 in `TASKS.md` Task 1.11).
- **Decision**:
  - Support both models cleanly via config/CLI (`index_<embedder>`).
  - Primary default for Phase 1 ingest tests: `google/siglip-base-patch16-224` to ensure batch stability without triggering GPU OOM or WDDM spill-to-host latency degradation on the 4 GB GPU.
  - Retain `google/siglip-so400m-patch14-384` with strict single-crop batching (`batch_size=1`) and text-tower CPU offloading as an available high-accuracy option when desktop windows are closed.

---

### [2026-10-08] DECISION-004: Three-Tier Pluggable Query Parser Interface
- **Status**: Adopted
- **Context**: Cloud LLMs (Gemini / Claude) offer flexible natural language extraction, but can fail due to missing keys, rate limits, or network interruptions.
- **Decision**: Implement a unified provider abstraction in `src/llm/providers.py`. Priority order: configured provider (`claude` or `gemini`) falling back to a deterministic, zero-dependency `rules` parser.
- **Alternatives Considered**: Hard-coding a single LLM SDK. Rejected to ensure graceful offline degradation and reproducible benchmark comparisons.

---

### [2026-10-08] DECISION-005: Cross-Camera Re-Identification via Temporal Constraint & Visual Cosine Sim
- **Status**: Adopted
- **Context**: Users need to trace entities moving across different camera fields of view without training an expensive domain-specific ReID neural network.
- **Decision**: Compute pairwise cosine similarities between SigLIP crop embeddings across tracks on different cameras, filtered by a configurable temporal gap window (`--max-gap-s`, default 7200s). Generate chronological trajectory reports (`cam A at <t1> -> cam B at <t2>`).
- **Alternatives Considered**: Training a separate ResNet/OSNet ReID model. Rejected due to dataset annotation requirements and GPU VRAM budget on 4 GB RTX 3050.

---

### [2026-10-08] DECISION-006: Standing Alert Rules & Event Triggers in SQLite WAL
- **Status**: Adopted
- **Context**: Real-time multi-camera operations require persistent alerting when specific objects (e.g., vehicles, pedestrians) enter monitored camera feeds.
- **Decision**: Maintain persistent `alert_rules` and `alert_events` tables directly in the primary SQLite WAL database. Pre-embed query text on rule creation/evaluation and vectorize candidate tracks with duplicate suppression to guarantee fast real-time online scanning.
- **Alternatives Considered**: External message brokers (Redis/RabbitMQ). Rejected to preserve self-contained, zero-external-dependency deployment architecture.

