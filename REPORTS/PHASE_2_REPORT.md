# Phase 2 Report: Query Engine & Search

## 1. Environment
- **Operating System**: Windows 11 (build 10.0.26100)
- **CPU / RAM**: Intel Core i5-13420H @ 2.10 GHz, 16 GB RAM
- **GPU / VRAM**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6141 MiB)
- **Python**: 3.13.7
- **PyTorch / CUDA**: torch 2.6.0+cu124, CUDA 12.4
- **Key Dependencies**: transformers 5.15.0, sentencepiece 0.2.2, ultralytics 8.3.82, OpenCV 4.11.0, pytest 9.1.1
- **FFmpeg**: ffmpeg version 9.0.2-full_build (Gyan.dev)

---

## 2. Acceptance Table

| Criterion / Requirement | Target / Constraint | Observed / Measured | Status |
|---|---|---|---|
| Rule-Based Query Parser | Extract visual prompt, location, time window without LLM | Offline rules regex parser ([src/query/parser.py](file:///c:/projects/MULTIStream/src/query/parser.py)) | **PASS** |
| Location Resolution | Resolve known cameras/aliases; flag unknown places as UNRESOLVED without guessing | "main gate" flagged `location_status: "UNRESOLVED"`, `resolved_camera: null` | **PASS** |
| Time Window Parsing | Support 8+ natural language temporal phrases ("Now" = latest indexed timestamp) | 8 test cases passing in `tests/test_parser.py` | **PASS** |
| Vector Similarity Search | SigLIP-Base cosine similarity retrieval over track & frame embeddings | Sub-2ms vector search over SQLite candidate embeddings | **PASS** |
| Timestamp & Offset Alignment | Snapshot, timestamp, offset, and cut clip refer to same moment | `tracks.t_best` and `tracks.offset_best` aligned to 2.336s / 18:00:02.335667 | **PASS** |
| On-Demand Clip Extraction | Window around event moment with padding | ffmpeg ultrafast cut, covers event moment, ffprobe duration=6.974s | **PASS** |
| Search Latency (Model Pre-loaded) | Sub-100ms total search without clip cutting | Parse=0.18ms, Embed=56.46ms, Search=1.23ms, Total(NoClip)=58.12ms | **PASS** |
| Evaluation Protocol | No tuning on held-out queries; check `eval/queries.json` | `eval/queries.json` DOES NOT EXIST; 0 held-out tuning | **PASS** |

---

## 3. Phase 2 Fixes & Raw Outputs

### Fix 1: Timestamp vs Offset Discrepancy & Alignment
**Explanation**:
In the initial Phase 2 run:
- `track_108`: `timestamp` was `2026-10-08T18:00:02.335667` while `offset_seconds` was `2.002` (difference ~0.334s).
- `track_454`: `timestamp` was `2026-10-08T18:00:09.175833` while `offset_seconds` was `9.009` (difference ~0.167s).

**Database Field Provenance**:
- `timestamp` originated from `tracks.t_best` (the timestamp of the highest-confidence/largest-area detection where the snapshot was taken).
- `offset_seconds` originally read `tracks.offset_start` (the offset of the first frame when ByteTrack acquired the track).
- Because physical objects are tracked across multiple frames before reaching their best detection frame, `offset_start` was earlier than `t_best` by 0.334s for track_108 and 0.167s for track_454.

**Fix Applied**:
1. Added column `offset_best REAL NOT NULL` to schema and database.
2. Updated [src/ingest/pipeline.py](file:///c:/projects/MULTIStream/src/ingest/pipeline.py) and [src/query/search.py](file:///c:/projects/MULTIStream/src/query/search.py) so `SearchResult.offset_seconds` reads `row["offset_best"]`.
3. Snapshot, timestamp (`t_best`), offset (`offset_best`), and clip extraction now all reference the exact same moment.

**Aligned Output (`scripts/query.py "a red car near cam_landscape" --top-k 1 --cut-clips --json`)**:
```json
{
  "query": "a red car near cam_landscape",
  "parsed": {
    "object_prompt": "a red car",
    "location": "cam_landscape",
    "location_status": "RESOLVED",
    "resolved_camera": "cam_landscape",
    "t_start": null,
    "t_end": null,
    "provider": "rules"
  },
  "latency_ms": 201.53,
  "results": [
    {
      "rank": 1,
      "id": "test_landscape_trk_108",
      "type": "track",
      "camera": "cam_landscape",
      "timestamp": "2026-10-08T18:00:02.335667",
      "offset_seconds": 2.336,
      "score": 0.1227,
      "label": "car",
      "video": "footage/test_landscape.mp4",
      "snapshot": "snapshots/test_landscape/track_108.jpg",
      "clip": "clips\\clip_test_landscape_trk_108.mp4"
    }
  ]
}
```

---

### Fix 2: Latency Benchmark (1 Warm-up + 10 Timed Runs)
**Configuration**:
- Script: [scripts/benchmark_latency.py](file:///c:/projects/MULTIStream/scripts/benchmark_latency.py)
- Query: `"a red car near cam_landscape"`
- Text model pre-loaded: **YES** (SigLIP loaded in GPU memory on `cuda:0` during `SearchEngine.__init__`)

**Raw Output**:
```text
Model loaded in 8.80 s. Text model already loaded in memory: YES.

--- Running 1 Warm-up Run ---
Warm-up complete.

--- Running 10 Timed Runs ---
Run  1: Parse=  0.19ms | Embed= 19.91ms | Search=  1.05ms | ClipCut=1793.92ms | Total(NoClip)= 21.15ms | Total(WithClip)=1815.08ms
Run  2: Parse=  0.19ms | Embed= 22.59ms | Search=  1.86ms | ClipCut=1667.82ms | Total(NoClip)= 24.64ms | Total(WithClip)=1692.46ms
Run  3: Parse=  0.20ms | Embed= 17.94ms | Search=  1.11ms | ClipCut=1826.34ms | Total(NoClip)= 19.26ms | Total(WithClip)=1845.59ms
Run  4: Parse=  0.19ms | Embed= 24.71ms | Search=  1.15ms | ClipCut=1721.94ms | Total(NoClip)= 26.05ms | Total(WithClip)=1747.98ms
Run  5: Parse=  0.17ms | Embed= 57.64ms | Search=  1.27ms | ClipCut=1628.97ms | Total(NoClip)= 59.08ms | Total(WithClip)=1688.05ms
Run  6: Parse=  0.16ms | Embed= 60.84ms | Search=  1.18ms | ClipCut=1633.72ms | Total(NoClip)= 62.19ms | Total(WithClip)=1695.91ms
Run  7: Parse=  0.19ms | Embed= 56.78ms | Search=  1.58ms | ClipCut=1766.09ms | Total(NoClip)= 58.58ms | Total(WithClip)=1824.67ms
Run  8: Parse=  0.17ms | Embed= 58.89ms | Search=  1.10ms | ClipCut=1715.62ms | Total(NoClip)= 60.16ms | Total(WithClip)=1775.78ms
Run  9: Parse=  0.18ms | Embed= 73.09ms | Search=  1.89ms | ClipCut=1637.34ms | Total(NoClip)= 75.16ms | Total(WithClip)=1712.50ms
Run 10: Parse=  0.18ms | Embed= 56.15ms | Search=  1.32ms | ClipCut=1840.52ms | Total(NoClip)= 57.65ms | Total(WithClip)=1898.17ms

================================================================================
LATENCY BENCHMARK RESULTS (N=10 runs after 1 warm-up)
================================================================================
Text model pre-loaded: YES (SigLIP on GPU cuda:0)
Stage 1 (Parse):       Median =   0.18 ms | Worst =   0.20 ms
Stage 2 (Embed):       Median =  56.46 ms | Worst =  73.09 ms
Stage 3 (Search):      Median =   1.23 ms | Worst =   1.89 ms
Stage 4 (Clip Cut):    Median = 1718.78 ms | Worst = 1840.52 ms
--------------------------------------------------------------------------------
Total WITHOUT Clip:    Median =  58.12 ms | Worst =  75.16 ms
Total WITH Clip Cut:   Median = 1761.88 ms | Worst = 1898.17 ms
================================================================================
```

---

### Fix 3: Time Parser 8 Cases & UNRESOLVED Tests
**Test Execution (`pytest tests/test_parser.py -v`)**:
Reference "Now" = `2026-10-08T19:00:00` (latest indexed timestamp).

**Raw Output**:
```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0 -- C:\Python313\python.exe
cachedir: .pytest_cache
rootdir: C:\projects\MULTIStream
configfile: pytest.ini
plugins: anyio-4.12.1
collecting ... collected 2 items

tests/test_parser.py::test_timeparse_8_cases PASSED                      [ 50%]
tests/test_parser.py::test_query_parser_rules_and_unresolved_location PASSED [100%]

============================= 2 passed in 12.57s ==============================
```

**Cases Tested**:
1. `last hour` -> `2026-10-08T18:00:00` to `2026-10-08T19:00:00`
2. `yesterday` -> `2026-10-07T00:00:00` to `2026-10-07T23:59:59.999999`
3. `between 2 and 3 PM` -> `2026-10-08T14:00:00` to `2026-10-08T15:00:00`
4. `this morning` -> `2026-10-08T06:00:00` to `2026-10-08T12:00:00`
5. `this afternoon` -> `2026-10-08T12:00:00` to `2026-10-08T18:00:00`
6. `past 30 minutes` -> `2026-10-08T18:30:00` to `2026-10-08T19:00:00`
7. `today` -> `2026-10-08T00:00:00` to `2026-10-08T19:00:00`
8. `in the last 2 hours` -> `2026-10-08T17:00:00` to `2026-10-08T19:00:00`

---

### Fix 4: ffprobe Cut Clip Verification
**Command**:
```powershell
ffprobe -v error -show_entries format=duration,start_time -of default=noprint_wrappers=1 clips/clip_test_landscape_trk_108.mp4
```
**Raw Output**:
```text
start_time=0.033367
duration=6.973633
```
**Coverage Verification**:
- Source video: `footage/test_landscape.mp4`.
- Track offset: `2.336 s`.
- Clip extraction window: starts at `max(0, 2.336 - 2.0) = 0.336 s`, duration ~6.97 s (spanning 0.336 s to ~7.31 s).
- The event moment at `2.336 s` falls at offset `2.000 s` within the extracted clip.

---

### Fix 5: Prompt Variants Comparison (Top-3 per Variant)
**Command**: `python scripts/compare_prompts.py`
**Raw Output**:
```text
==========================================================================================
PROMPT VARIANT COMPARISON (TOP-3 RESULTS PER VARIANT)
==========================================================================================

Query: 'red car' -> Parsed Object Prompt: 'a red car'
------------------------------------------------------------------------------------------
  Rank 1: ID=test_landscape_trk_108    Cam=cam_landscape    Label=car      Timestamp=2026-10-08T18:00:02.335667 Offset=2.336 s Score=0.1227
  Rank 2: ID=test_landscape_trk_675    Cam=cam_landscape    Label=car      Timestamp=2026-10-08T18:00:14.681333 Offset=14.681s Score=0.0676
  Rank 3: ID=test_landscape_trk_454    Cam=cam_landscape    Label=truck    Timestamp=2026-10-08T18:00:09.175833 Offset=9.176 s Score=0.0591

Query: 'a photo of a red car' -> Parsed Object Prompt: 'a photo of a red car'
------------------------------------------------------------------------------------------
  Rank 1: ID=test_landscape_trk_108    Cam=cam_landscape    Label=car      Timestamp=2026-10-08T18:00:02.335667 Offset=2.336 s Score=0.1304
  Rank 2: ID=test_landscape_trk_675    Cam=cam_landscape    Label=car      Timestamp=2026-10-08T18:00:14.681333 Offset=14.681s Score=0.0844
  Rank 3: ID=test_landscape_trk_384    Cam=cam_landscape    Label=car      Timestamp=2026-10-08T18:00:07.841167 Offset=7.841 s Score=0.0686

Query: 'a car' -> Parsed Object Prompt: 'a car'
------------------------------------------------------------------------------------------
  Rank 1: ID=test_landscape_trk_108    Cam=cam_landscape    Label=car      Timestamp=2026-10-08T18:00:02.335667 Offset=2.336 s Score=0.0857
  Rank 2: ID=test_video03_trk_41       Cam=test_video03     Label=car      Timestamp=2026-10-08T15:07:42.039141 Offset=7.666 s Score=0.0852
  Rank 3: ID=test_video03_trk_70       Cam=test_video03     Label=car      Timestamp=2026-10-08T15:07:49.371719 Offset=14.998s Score=0.0820
==========================================================================================
```

---

### Fix 6: Unknown Place Marked as UNRESOLVED
**Query**: `"a person at the main gate"`
**Raw Output (`python scripts/query.py "a person at the main gate" --json`)**:
```json
{
  "query": "a person at the main gate",
  "parsed": {
    "object_prompt": "a person",
    "location": "main gate",
    "location_status": "UNRESOLVED",
    "resolved_camera": null,
    "t_start": null,
    "t_end": null,
    "provider": "rules"
  },
  "latency_ms": 648.85,
  "results": []
}
```
"main gate" is explicitly marked as `UNRESOLVED` and `resolved_camera: null`. It is not guessed or mapped to any existing camera.

---

### Fix 7: Evaluation Queries & Held-out Split Status
- `eval/queries.json` exists: **NO** (File does not exist in repository).
- Tuning used held-out queries: **NO**. No evaluation set exists yet and 0 tuning has been performed against any held-out split.

---

## 4. Deviations
- None. Offline rule parser implemented without external APIs or LLMs. Clip extraction uses ffmpeg directly with on-demand execution.

---

## 5. Known Issues
- `clips/` directory generation requires `ffmpeg` available on system PATH or via WinGet fallback.
- First invocation without model preloading incurs an initial ~8.8s PyTorch/CUDA weights load overhead. Subsequent queries with preloaded models execute search in ~58 ms median.

---

## 6. What was NOT Verified
- Natural language parsing through external LLM providers (Gemini, Claude, OpenAI): offline rule parser used exclusively per instructions.
- Held-out retrieval benchmark: `eval/queries.json` does not exist; formal evaluation deferred to Phase 5.

---

## 7. Git Log
*(Appended upon commit)*
