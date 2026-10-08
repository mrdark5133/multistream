# Phase 1 Report: Ingest Pipeline
Status: PASSED
Date/time: 2026-10-08T17:48:00+05:30
Parent commit: 99de46c7576a16c7cf6dd29c4202fbff8eb76e63

## 1. Summary
Phase 1 implemented and verified the complete multi-camera ingestion pipeline:
- SQLite database schema in WAL mode with `config`, `videos`, `tracks`, `frames`, and `aliases` tables ([src/index/schema.sql](file:///c:/projects/MULTIStream/src/index/schema.sql), [src/index/db.py](file:///c:/projects/MULTIStream/src/index/db.py)).
- 4-stage start time resolution fallback chain (`manifest` -> `filename` -> `ffprobe` -> `mtime_fallback`) with rotation extraction ([src/ingest/start_time.py](file:///c:/projects/MULTIStream/src/ingest/start_time.py)).
- YOLO-World detection (`yolov8s-worldv2.pt`, `vocab.yaml`) + ByteTrack tracking with configurable stride (default: 5), best-frame snapshot selection ($A \times c$), and strict multi-video tracker isolation ([src/ingest/detect_track.py](file:///c:/projects/MULTIStream/src/ingest/detect_track.py)).
- `google/siglip-base-patch16-224` image and text embeddings with automatic OOM batch backoff and $L_2$ unit normalization ([src/ingest/embed.py](file:///c:/projects/MULTIStream/src/ingest/embed.py)).
- Complete CLI ingestion tool `scripts/ingest.py` with incremental duplicate prevention and structured logging ([scripts/ingest.py](file:///c:/projects/MULTIStream/scripts/ingest.py)).
- Verified image tower latency benchmark across batch sizes 1/8/16 comparing SigLIP-Base against SigLIP-SO400M.
- 15/15 unit and integration tests passing in pytest.

---

## 2. Raw Test Outputs

### Key Test 1: Full Automated Test Suite (`pytest -v`)
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\projects\MULTIStream\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\projects\MULTIStream
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 15 items

tests/test_db.py::test_wal_mode_and_foreign_keys PASSED                  [  6%]
tests/test_db.py::test_config_table PASSED                               [ 13%]
tests/test_db.py::test_embedding_blob_roundtrip PASSED                   [ 20%]
tests/test_db.py::test_video_and_track_fk PASSED                         [ 26%]
tests/test_embed.py::test_embed_images_shape_and_norm PASSED             [ 33%]
tests/test_embed.py::test_embed_text_shape_and_norm PASSED               [ 40%]
tests/test_embed.py::test_embed_empty PASSED                             [ 46%]
tests/test_start_time.py::test_parse_filename_time PASSED                [ 53%]
tests/test_start_time.py::test_manual_override PASSED                    [ 60%]
tests/test_start_time.py::test_manifest_resolution PASSED                [ 66%]
tests/test_start_time.py::test_filename_fallback PASSED                  [ 73%]
tests/test_start_time.py::test_mtime_fallback PASSED                     [ 80%]
tests/test_start_time.py::test_parse_rotation PASSED                     [ 86%]
tests/test_tracker_isolation.py::test_reset_tracker_resets_basetrack_and_active_tracks PASSED [ 93%]
tests/test_tracker_isolation.py::test_multi_video_tracker_isolation PASSED [100%]

======================== 15 passed in 72.73s (0:01:12) ========================
```

---

### Key Test 2: Task 1.11 Image Tower Benchmark (`scripts/benchmark_embedders.py`)
Timed across 3 warm-up + 20 timed runs on real upright crops (`test_video01-03.mp4`):
```text
==========================================================================================
TASK 1.11 SUMMARY BENCHMARK TABLE: IMAGE TOWER ONLY
==========================================================================================
Model                            | BS  | Mean (ms) | Median (ms) | Min (ms) | Max (ms) | Crops/s 
------------------------------------------------------------------------------------------
SigLIP-Base (patch16-224)        | 1   | 18.96     | 12.92       | 11.64    | 44.52    | 52.7    
SigLIP-Base (patch16-224)        | 8   | 29.17     | 29.13       | 28.66    | 29.84    | 274.2   
SigLIP-Base (patch16-224)        | 16  | 54.74     | 54.64       | 54.11    | 55.45    | 292.3   
------------------------------------------------------------------------------------------
SigLIP-SO400M (patch14-384)      | 1   | 67.12     | 67.10       | 66.04    | 68.08    | 14.9    
SigLIP-SO400M (patch14-384)      | 8   | 498.77    | 498.74      | 496.53   | 500.42   | 16.0    
SigLIP-SO400M (patch14-384)      | 16  | 999.27    | 998.91      | 995.91   | 1002.67  | 16.0    
==========================================================================================
```
- **SigLIP-Base Peak Inference VRAM**: 1438 MiB
- **SigLIP-SO400M Peak Inference VRAM**: 3324 MiB

---

### Key Test 3: CLI Batch Ingestion & Duplicate Prevention (`scripts/ingest.py`)
```text
================================================================================
MULTIStream Ingestion Pipeline
Index DB:      index_base\index.db
Snapshots:     index_base\snapshots
Embedder:      google/siglip-base-patch16-224
Stride:        5
Imgsz:         640
================================================================================
Found 5 video(s) for ingestion.

--- Ingesting: footage\test_landscape.mp4 ---
  [INDEXED] Camera:        test_landscape
            Start Time:    2022-06-21T18:02:49+00:00 (source: ffprobe)
            Duration:      15.08s | Rotation: 0 deg
            Tracks Extr:   62
            Frames Extr:   8
            Wall Time:     10.98s
            VRAM Used:     2702 MiB

--- Ingesting: footage\test_landscape2.mp4 ---
  [INDEXED] Camera:        test_landscape2
            Start Time:    2022-08-17T13:52:26+00:00 (source: ffprobe)
            Duration:      29.00s | Rotation: 0 deg
            Tracks Extr:   29
            Frames Extr:   15
            Wall Time:     35.77s
            VRAM Used:     2642 MiB

--- Ingesting: footage\test_video01.mp4 ---
  [INDEXED] Camera:        test_video01
            Start Time:    2026-10-08T15:07:05.287875 (source: mtime_fallback)
            Duration:      4.27s | Rotation: 0 deg
            Tracks Extr:   7
            Frames Extr:   3
            Wall Time:     1.27s
            VRAM Used:     2638 MiB

--- Ingesting: footage\test_video02.mp4 ---
  [INDEXED] Camera:        test_video02
            Start Time:    2026-10-08T15:07:22.734026 (source: mtime_fallback)
            Duration:      9.13s | Rotation: 0 deg
            Tracks Extr:   4
            Frames Extr:   5
            Wall Time:     1.83s
            VRAM Used:     2644 MiB

--- Ingesting: footage\test_video03.mp4 ---
  [INDEXED] Camera:        test_video03
            Start Time:    2026-10-08T15:07:34.373265 (source: mtime_fallback)
            Duration:      16.16s | Rotation: 0 deg
            Tracks Extr:   12
            Frames Extr:   8
            Wall Time:     3.39s
            VRAM Used:     2640 MiB

================================================================================
INGESTION SUMMARY
================================================================================
Total videos processed: 5
Total tracks indexed:   114
Total frames indexed:   39
Total ingest wall time: 53.24s
================================================================================
```

Duplicate prevention verification:
```text
--- Ingesting: footage\test_video01.mp4 ---
[Duplicate Prevention] Skipping footage/test_video01.mp4: already indexed at 2026-10-08T17:39:23.317684
  [SKIPPED] Video already indexed at 2026-10-08T17:39:23.317684
```

---

## 3. Database State Verification (`index_base/index.db`)
```sql
SELECT COUNT(*) FROM videos; -- 5 videos
SELECT COUNT(*) FROM tracks; -- 114 tracks (with 768-dim FP32 BLOB embeddings + JPEG snapshots)
SELECT COUNT(*) FROM frames; -- 39 whole frames (with 768-dim FP32 BLOB embeddings + JPEG snapshots)
```

---

## 4. What was NOT verified / Deviations
- Held-out evaluation queries (`eval/queries.json`): human ground truth pending Phase 5.
- Online LLM query parsing: offline ingestion only, LLM parser begins in Phase 2.
