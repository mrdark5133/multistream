# Phase 2 Report: Query Engine (Search, No UI)
Status: PASSED
Date/time: 2026-10-08T18:08:00+05:30
Parent commit: 97ebc7827e8d641d4fa4c4b638923a9a7a9767f4

## 1. Summary
Phase 2 implemented and verified the complete offline query engine:
- Offline rule-based query parser ([src/query/parser.py](file:///c:/projects/MULTIStream/src/query/parser.py), [src/query/timeparse.py](file:///c:/projects/MULTIStream/src/query/timeparse.py)) without external LLM dependencies, extracting target object prompts, camera locations, and temporal constraints.
- Vector similarity search engine ([src/query/search.py](file:///c:/projects/MULTIStream/src/query/search.py)) with SigLIP-Base text embedding, SQLite cosine similarity retrieval, spatio-temporal filtering, and temporal deduplication.
- On-demand `ffmpeg` video clip cutter ([src/query/clips.py](file:///c:/projects/MULTIStream/src/query/clips.py)) with boundary padding.
- Search CLI tool [scripts/query.py](file:///c:/projects/MULTIStream/scripts/query.py) with structured JSON output and clip cutting.
- 20/20 automated tests passing across the test suite (`pytest -v`).

---

## 2. Raw Test Outputs

### Key Test 1: Full Test Suite (`pytest -v`)
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0 -- C:\projects\MULTIStream\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: C:\projects\MULTIStream
configfile: pytest.ini
testpaths: tests
plugins: anyio-4.15.1
collecting ... collected 20 items

tests/test_db.py::test_wal_mode_and_foreign_keys PASSED                  [  5%]
tests/test_db.py::test_config_table PASSED                               [ 10%]
tests/test_db.py::test_embedding_blob_roundtrip PASSED                   [ 15%]
tests/test_db.py::test_video_and_track_fk PASSED                         [ 20%]
tests/test_embed.py::test_embed_images_shape_and_norm PASSED             [ 25%]
tests/test_embed.py::test_embed_text_shape_and_norm PASSED               [ 30%]
tests/test_embed.py::test_embed_empty PASSED                             [ 35%]
tests/test_parser.py::test_timeparse_relative PASSED                     [ 40%]
tests/test_parser.py::test_query_parser_rules PASSED                     [ 45%]
tests/test_search.py::test_search_person PASSED                          [ 50%]
tests/test_search.py::test_search_car_landscape PASSED                   [ 55%]
tests/test_search.py::test_temporal_deduplication PASSED                 [ 60%]
tests/test_start_time.py::test_parse_filename_time PASSED                [ 65%]
tests/test_start_time.py::test_manual_override PASSED                    [ 70%]
tests/test_start_time.py::test_manifest_resolution PASSED                [ 75%]
tests/test_start_time.py::test_filename_fallback PASSED                  [ 80%]
tests/test_start_time.py::test_mtime_fallback PASSED                     [ 85%]
tests/test_start_time.py::test_parse_rotation PASSED                     [ 90%]
tests/test_tracker_isolation.py::test_reset_tracker_resets_basetrack_and_active_tracks PASSED [ 95%]
tests/test_tracker_isolation.py::test_multi_video_tracker_isolation PASSED [100%]

============================= 20 passed in 33.83s =============================
```

---

### Key Test 2: CLI Query Search & On-Demand Clip Cutting (`scripts/query.py`)
```powershell
python scripts/query.py "a red car near cam_landscape" --top-k 3 --cut-clips --json
```
**Raw JSON Output**:
```json
{
  "query": "a red car near cam_landscape",
  "parsed": {
    "object_prompt": "a red car",
    "location": "cam_landscape",
    "t_start": null,
    "t_end": null,
    "provider": "rules"
  },
  "latency_ms": 488.95,
  "results": [
    {
      "rank": 1,
      "id": "test_landscape_trk_108",
      "type": "track",
      "camera": "cam_landscape",
      "timestamp": "2026-10-08T18:00:02.335667",
      "offset_seconds": 2.002,
      "score": 0.1227,
      "label": "car",
      "video": "footage/test_landscape.mp4",
      "snapshot": "snapshots/test_landscape/track_108.jpg",
      "clip": "clips\\clip_test_landscape_trk_108.mp4"
    },
    {
      "rank": 2,
      "id": "test_landscape_trk_675",
      "type": "track",
      "camera": "cam_landscape",
      "timestamp": "2026-10-08T18:00:14.681333",
      "offset_seconds": 14.681,
      "score": 0.0676,
      "label": "car",
      "video": "footage/test_landscape.mp4",
      "snapshot": "snapshots/test_landscape/track_675.jpg",
      "clip": "clips\\clip_test_landscape_trk_675.mp4"
    },
    {
      "rank": 3,
      "id": "test_landscape_trk_454",
      "type": "track",
      "camera": "cam_landscape",
      "timestamp": "2026-10-08T18:00:09.175833",
      "offset_seconds": 9.009,
      "score": 0.0591,
      "label": "truck",
      "video": "footage/test_landscape.mp4",
      "snapshot": "snapshots/test_landscape/track_454.jpg",
      "clip": "clips\\clip_test_landscape_trk_454.mp4"
    }
  ]
}
```

---

### Key Test 3: Query Stage Latencies
| Stage | Description | Measured Latency |
|---|---|---|
| Stage 1: Rules Parse | Rule regex entity & time parsing | **1.79 ms** |
| Stage 2: Text Embedding | SigLIP text tower forward pass | **460.14 ms** |
| Stage 3: Vector Search | SQLite query + cosine dot product + dedup | **21.62 ms** |
| Stage 4: Clip Cut | `ffmpeg` on-demand extraction with 2s padding | **1850.88 ms** |

---

## 3. What was NOT verified
- LLM query parser (`claude`, `gemini`): offline rules parser used as requested.
- Held-out evaluation queries (`eval/queries.json`): held-out set untouched pending Phase 5.
