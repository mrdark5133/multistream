# Evaluation Protocol & Benchmarking Specification

## 1. Ground Truth Format (`eval/queries.json`)
The evaluation query set is supplied and curated by the human team. The build agent does not write ground truth labels or tune on the held-out split.

Structure:
```json
[
  {
    "id": "q1",
    "query": "red car at the gate in the last hour",
    "split": "dev", // or "held-out"
    "truth": [
      {
        "camera": "cam_gate",
        "start": "2026-10-08T11:40:00",
        "end": "2026-10-08T11:45:00"
      }
    ]
  }
]
```

## 2. Evaluation Metrics

### 2.1 Hit Definition
A retrieved candidate $R = (c_R, t_R)$ from top-$K$ results is counted as a **Hit** if:
1. **Camera Match**: $c_R == c_{truth}$
2. **Temporal Localization Match**: The retrieved timestamp $t_R$ falls within the ground truth interval $[start_{truth} - \Delta, end_{truth} + \Delta]$, where default tolerance $\Delta = 5\text{ seconds}$.

### 2.2 Primary Metrics
- **Top-1 Accuracy**: Proportion of queries where the rank-1 result is a Hit.
- **Top-5 Accuracy**: Proportion of queries where at least one of the top-5 results is a Hit.
- **Camera-Only Accuracy (Top-1)**: Camera correctly identified regardless of timestamp match.
- **Time-Localization Accuracy (Top-1)**: Timestamp matched within tolerance given that camera matched.
- **Parse Accuracy**: Percentage of test queries parsing into valid schema JSON with expected fields.
- **Query Latency**: Measured in milliseconds (Median, P95, and Worst over 5 repetitions), broken down by stage:
  - Query parse latency
  - Text embedding latency
  - Vector similarity search latency
  - Video clip extraction latency

## 3. Baseline Definition
The official baseline represents a conventional whole-frame retrieval approach:
- Sample whole video frames at fixed interval (1 fps).
- Extract SigLIP whole-frame embeddings (same embedder model as proposed pipeline).
- No object detection (no YOLO-World).
- No multi-object tracking (no ByteTrack).
- No crop extraction.
- At query time: vector search directly over whole-frame embeddings.

## 4. Ablation Study Rows
All ablations will be evaluated against the exact same test queries:
1. **Baseline**: Whole-frame retrieval (1 fps, no detection, no tracking).
2. **+ Detection Crops**: YOLO-World detection on raw sampled frames, embedding every raw detected bounding box crop (no tracking).
3. **+ Tracking (Per-Track Embedding)**: YOLO-World + ByteTrack, embedding only the single best-frame crop per track.
4. **+ Extended Vocabulary**: Expanded CCTV vocabulary (~60 classes vs 10 base classes).
5. **+ Whole-Frame Fallback**: Combined track crop search + periodic whole-frame search (merged results).
6. **+ Temporal Deduplication**: Merging temporally proximate hits on the same camera.
7. **+ VLM Reranking (Phase 6)**: Multimodal reranking of top-10 candidate snapshots.

## 5. Storage & Operational Efficiency Metrics
- Total indexing wall time vs raw video duration (Real-Time Factor / Ingest Speed Ratio).
- Peak VRAM during indexing (MB).
- Peak VRAM during query serving (MB).
- Disk footprint of SQLite index and snapshot evidence cache (MB).
