# Phase 3 Report: Clarify-Once Memory

## 1. Environment
- **Operating System**: Windows 11 (build 10.0.26100)
- **CPU / RAM**: Intel Core i5-13420H @ 2.10 GHz, 16 GB RAM
- **GPU / VRAM**: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6141 MiB)
- **Python**: 3.13.7
- **PyTorch / CUDA**: torch 2.6.0+cu124, CUDA 12.4
- **Key Dependencies**: transformers 5.15.0, sentencepiece 0.2.2, ultralytics 8.3.82, OpenCV 4.11.0, pytest 9.1.1
- **SQLite**: 3.45.1 (WAL mode enabled)

---

## 2. Acceptance Table

| Test / Requirement | Command / Trigger | Expected | Actual | Status |
|---|---|---|---|---|
| **3.4.a Unknown Referent Clarification** | `engine.query("a person at the main gate")` | Status `clarify`, `referent: "main gate"`, options list | Returned `status: "clarify"`, `referent: "main gate"`, 5 camera options | **PASS** |
| **3.4.b Stored Alias Persistence Across Paraphrases** | `save_alias("main gate", "cam_landscape")` then query paraphrases | Resolves to `cam_landscape` without asking | "at the main gate", "near main gate", "by the main gate" all resolve without asking | **PASS** |
| **3.4.c Cross-Process Restart Persistence** | Save in process 1, query in fresh process 2 | Resolves from SQLite table across restarts | Fresh instance loaded alias and resolved without clarification | **PASS** |
| **3.4.d Known Camera Direct Bypass** | `engine.query("a red car near cam_landscape")` | Direct bypass of clarification | Resolved to `cam_landscape`, status `success`, results returned | **PASS** |
| **3.4.e Polygon Spatial Filtering** | Point-in-polygon ray-casting on normalized track centers | Retain only tracks whose center is inside polygon | Track at (0.2, 0.2) retained; track at (0.8, 0.8) excluded | **PASS** |
| **3.5 Disallow Guessing with Tempting Context** | Query "main gate" when DB has `cam_gate_2` | Must NOT guess `cam_gate_2` | Returned `status: "clarify"`, 0 guesses | **PASS** |
| **CLI Clarify & Alias Management** | `scripts/query.py --save-alias ...` | Store alias and resolve queries | Stored alias via CLI and resolved subsequent queries | **PASS** |

---

## 3. Commands Run and Raw Output

### Key Test 1: Full Phase 3 Automated Test Suite (`pytest tests/test_memory.py -v`)
```text
============================= test session starts =============================
platform win32 -- Python 3.13.7, pytest-9.1.1, pluggy-1.6.0 -- C:\Python313\python.exe
cachedir: .pytest_cache
rootdir: C:\projects\MULTIStream
configfile: pytest.ini
plugins: anyio-4.12.1
collecting ... collected 8 items

tests/test_memory.py::test_alias_normalization PASSED                    [ 12%]
tests/test_memory.py::test_point_in_polygon_geometry PASSED              [ 25%]
tests/test_memory.py::test_unknown_referent_triggers_clarification PASSED [ 37%]
tests/test_memory.py::test_stored_alias_resolves_without_asking_across_paraphrases PASSED [ 50%]
tests/test_memory.py::test_cross_process_restart_persistence PASSED      [ 62%]
tests/test_memory.py::test_known_camera_names_bypass_clarification PASSED [ 75%]
tests/test_memory.py::test_polygon_spatial_filtering PASSED              [ 87%]
tests/test_memory.py::test_disallow_guessing_with_tempting_context PASSED [100%]

======================== 8 passed in 63.22s (0:01:03) =========================
```

---

### Key Test 2: CLI Clarify Request on Unknown Referent
**Command**:
```powershell
python scripts/query.py "a person at the main gate" --json
```
**Raw Output**:
```json
{
  "status": "clarify",
  "referent": "main gate",
  "options": [
    "cam_landscape",
    "cam_landscape2",
    "test_video01",
    "test_video02",
    "test_video03"
  ]
}
```

---

### Key Test 3: CLI Alias Registration
**Command**:
```powershell
python scripts/query.py --save-alias "main gate" --camera "cam_landscape" --json
```
**Raw Output**:
```json
{
  "status": "saved",
  "name": "main gate",
  "camera": "cam_landscape",
  "polygon": null
}
```

---

### Key Test 4: CLI Query Resolution After Alias Saved (No Clarification)
**Command**:
```powershell
python scripts/query.py "a person at the main gate" --top-k 1 --json
```
**Raw Output**:
```json
{
  "query": "a person at the main gate",
  "parsed": {
    "object_prompt": "a person",
    "location": "main gate",
    "location_status": "RESOLVED",
    "resolved_camera": "cam_landscape",
    "t_start": null,
    "t_end": null,
    "provider": "rules"
  },
  "latency_ms": 199.16,
  "results": [
    {
      "rank": 1,
      "id": "test_landscape_trk_11",
      "type": "track",
      "camera": "cam_landscape",
      "timestamp": "2026-10-08T18:00:00",
      "offset_seconds": 0.0,
      "score": 0.06,
      "label": "pedestrian",
      "video": "footage/test_landscape.mp4",
      "snapshot": "snapshots/test_landscape/track_11.jpg",
      "clip": null
    }
  ]
}
```

---

## 4. Deviations
- None. `aliases` store persists to SQLite `aliases` table with normalized name matching, ray-casting point-in-polygon filtering, and zero LLM guessing.

---

## 5. Known Issues
- Very complex concave polygons with self-intersections should be validated prior to storage (standard ray-casting supports arbitrary simple polygons).

---

## 6. What was NOT Verified
- Interactive browser canvas polygon drawing: verified via automated coordinate polygon tests and CLI JSON argument; interactive canvas will be verified in Phase 4 Chat UI.

---

## 7. Git Log
```text
ca88c9e | irfanbasha11012007-max <irfanbasha11012007@gmail.com> | phase3: implement clarify-once memory, polygon filtering, and test suite
d67ed5f | irfanbasha11012007-max <irfanbasha11012007@gmail.com> | phase2: query engine fixes, offset alignment, latency benchmark, and report
607e67c | harivarman-007 <harivarman124@gmail.com> | phase2: implement offline rules parser, vector search, clip extraction, and query cli
```
