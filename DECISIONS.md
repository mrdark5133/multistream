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
- **Status**: Measured & Decided
- **Context**: Profiled YOLO-World (`yolov8s-worldv2.pt`) co-loaded with both SigLIP models in FP16 on real footage (`footage/test_gate.mp4` frame).
- **Empirical Measurements**:
  1. **`google/siglip-so400m-patch14-384` (FP16)**:
     - Peak PyTorch Allocated: **2379.7 MB**
     - Total System VRAM Used: **3647.0 MB** / 4094.0 MB
     - Headroom Remaining: **246.0 MB** (6.0% buffer)
     - Single Crop Inference: **464.1 ms**
     - Headroom risk: Batch size must be restricted to 1 during ingest to avoid CUDA OOM.
  2. **`google/siglip-base-patch16-224` (FP16)**:
     - Peak PyTorch Allocated: **1086.2 MB**
     - Total System VRAM Used: **2279.0 MB** / 4094.0 MB
     - Headroom Remaining: **1614.0 MB** (39.4% buffer)
     - Single Crop Inference: **198.6 ms** (2.34x faster)
- **Decision**:
  - Support both models cleanly via config/CLI (`index_<embedder>`).
  - Primary default for Phase 1 ingest tests: `google/siglip-base-patch16-224` to ensure batch stability without triggering GPU OOM crashes on the 4 GB GPU.
  - Retain `google/siglip-so400m-patch14-384` with strict single-crop batching (`batch_size=1`) and text-tower CPU offloading as an available high-accuracy option.

---

### [2026-10-08] DECISION-004: Three-Tier Pluggable Query Parser Interface
- **Status**: Adopted
- **Context**: Cloud LLMs (Gemini / Claude) offer flexible natural language extraction, but can fail due to missing keys, rate limits, or network interruptions.
- **Decision**: Implement a unified provider abstraction in `src/llm/providers.py`. Priority order: configured provider (`claude` or `gemini`) falling back to a deterministic, zero-dependency `rules` parser.
- **Alternatives Considered**: Hard-coding a single LLM SDK. Rejected to ensure graceful offline degradation and reproducible benchmark comparisons.
