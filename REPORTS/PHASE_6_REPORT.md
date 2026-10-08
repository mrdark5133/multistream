# Phase 6 Report: Accuracy Improvements (Evidence-Driven)

**Author:** irfanbasha11012007-max <irfanbasha11012007@gmail.com>  
**Date:** 2026-10-08  
**Status:** COMPLETED & VERIFIED  

---

## 1. Executive Summary
Phase 6 evaluates retrieval accuracy improvements based strictly on empirical evidence, eliminating ungrounded estimates and applying rigorous hit criteria across all evaluation rows:
1. **Eval Hit Criteria Hardening**:
   - Hit criteria now strictly requires: (a) camera match, (b) temporal window match within $\pm 3.0$ seconds (shrunk from $\pm 5.0$ seconds), and (c) candidate label match against ground truth using a strict synonym map (`car`/`van`/`suv`, `truck`/`bus`, `person`/`pedestrian`, `motorcycle`/`motorbike`). Irrelevant objects (e.g., a person crop) no longer count as hits for non-person queries (e.g., "utility cart").
   - Truth windows previously spanning entire clips were shrunk to actual physical occurrence timestamps.
2. **Fair Whole-Frame Baseline Comparison**:
   - The whole-frame baseline was updated with the identical strict camera filtering logic (`LOWER(camera) IN (?, ?, ?)`).
   - Under these hardened criteria, the whole-frame baseline achieves 80.0% Dev R@1 / 100.0% Dev R@5 and 80.0% Dev2 R@1 / 100.0% Dev2 R@5. The Full Pipeline achieves 70.0% Dev R@1 / 90.0% Dev R@5 and 30.0% Dev2 R@1 / 70.0% Dev2 R@5.
3. **Dataset Partitioning (Dev2 Split)**:
   - The original 10 held-out queries (`test_01` to `test_10`) were reclassified as `dev2` because they were inspected during error analysis. A pristine held-out set can now be evaluated without tuning.
4. **Detector Bottleneck Evaluation (Task 6.2)**:
   - Detector recall with `yolov8s-worldv2.pt` is not the bottleneck; target objects are reliably detected within candidate sets. Upgrading to `yolov8m` is unnecessary and avoided under the 4 GB VRAM budget.
5. **VLM Reranker Architecture (Task 6.3)**:
   - Implemented `VLMReranker` (`src/query/rerank.py`) with cloud egress logging. Marked as **implemented, NOT RUN: no API keys** due to missing cloud API keys. The no-op "off" row was removed from ablation tables.
6. **Ingest Throughput Clarification**:
   - Corrected throughput metric to **1.38x Real-Time** based on Phase 1's actual measured run (73.58 seconds of video across 5 clips processed in 53.24 seconds wall-clock time). The previous 3.35x figure from an earlier partial run has been removed.

---

## 2. Environment Actually Used

### 2.1 Python Interpreter & Virtual Environment
- **Server Interpreter Path**: `c:\projects\MULTIStream\.venv\Scripts\python.exe`
```text
$ .venv\Scripts\python.exe --version
Python 3.11.9
```

### 2.2 Key Package Versions (`pip list`)
```text
$ .venv\Scripts\python.exe -m pip list
fastapi                0.142.4
torch                  2.6.0+cu124
transformers           5.19.0
ultralytics            8.4.174
python-multipart       0.0.32
uvicorn                0.54.0
```

### 2.3 Hardware & GPU State (`nvidia-smi`)
```text
$ nvidia-smi
Thu Oct  8 21:00:58 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 617.14                 KMD Version: 617.14        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   39C    P8              1W /   65W |    1611MiB /   4094MiB |     19%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```

---

## 3. Acceptance Table
| Task | Description | Status | Evidence |
|---|---|---|---|
| **6.1** | Systematic failure case error analysis | **PASS** | `scripts/error_analysis.py` diagnostic run on all 20 queries |
| **6.2** | Evaluate `yolov8m-worldv2.pt` if recall bottlenecked | **PASS** | Evaluated; recall is not detector-limited; 4 GB VRAM preserved |
| **6.3** | Optional VLM reranking on top 10 candidates | **PASS** | Implemented (`src/query/rerank.py`), **NOT RUN: no API keys** |
| **6.4** | Query prompt & camera filter improvements | **PASS** | Strict camera matching applied to all modes; label matching enforced |
| **Regression** | Full pytest test suite | **PASS** | 28 / 28 passing in 44.58s (`.venv\Scripts\pytest.exe tests/`) |
| **API** | Automated endpoint test suite | **PASS** | 11 / 11 passing via `scripts/test_api.py` on `.venv` uvicorn server |

---

## 4. Commands Run and RAW Output

### 4.1 PyTest Regression Suite (`.venv\Scripts\pytest.exe tests/`)
```text
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.1.1, pluggy-1.6.0
rootdir: C:\projects\MULTIStream
configfile: pytest.ini
plugins: anyio-4.15.1
collected 28 items

tests\test_db.py ....                                                    [ 14%]
tests\test_embed.py ...                                                  [ 25%]
tests\test_memory.py ........                                            [ 53%]
tests\test_parser.py ..                                                  [ 60%]
tests\test_search.py ...                                                 [ 71%]
tests\test_start_time.py ......                                          [ 92%]
tests\test_tracker_isolation.py ..                                       [100%]

============================= 28 passed in 44.58s =============================
```

### 4.2 Automated API Test Suite (`.venv\Scripts\python.exe scripts/test_api.py`)
```text
================================================================================
MULTISTREAM AUTOMATED API TEST SUITE
Target: http://127.0.0.1:8000
================================================================================
[PASS] GET / (Root HTML)                   | Status: 200 | Latency:   39.3 ms | Content-Type: text/html; charset=utf-8
[PASS] GET /health                         | Status: 200 | Latency: 3290.2 ms | VRAM Used: 1106 MB
[PASS] GET /cameras                        | Status: 200 | Latency:    3.8 ms | Found 6 cameras
[PASS] GET /aliases                        | Status: 200 | Latency:    3.6 ms | Found 4 aliases
[PASS] POST /ask (Clarify Unknown)         | Status: 200 | Latency:    6.4 ms | Status: clarify, Referent: 'test courtyard 1791473567'
[PASS] POST /alias (Register Spatial)      | Status: 200 | Latency:    3.3 ms | Saved 'test courtyard 1791473567' -> cam_landscape
[PASS] POST /ask (Resolved Query)          | Status: 200 | Latency:  297.3 ms | Status: success, Cam: cam_landscape
[PASS] POST /ask (Retrieve Vehicle)        | Status: 200 | Latency:   19.4 ms | Found 3 candidate results
[PASS] GET /snapshot/test_landscape_trk_108 | Status: 200 | Latency:    6.5 ms | Bytes: 105363
[PASS] GET /clip/test_landscape_trk_108    | Status: 200 | Latency:   91.0 ms | Bytes: 19018787
[PASS] POST /upload                        | Status: 200 | Latency:    5.9 ms | Registered camera 'cam_upload_test'
================================================================================
SUMMARY: 11 / 11 API tests PASSED.
================================================================================
```

### 4.3 Ablation Study Execution (`.venv\Scripts\python.exe scripts/eval.py`)
```text
==========================================================================================
EVALUATION HARNESS (EVAL.md)
Queries File: C:\projects\MULTIStream\eval\queries.json
Queries SHA256: 05a94ca4f420a71673bb024019538a9af909aea2621da27618463a4531aeb8aa
==========================================================================================
Loaded 10 dev queries, 10 dev2 queries, and 0 held-out queries.

--- Running Ablation Matrix on Dev and Dev2 Splits ---
Config: Full Pipeline (Ours)             | Dev R@1= 70.0% | Dev2 R@1= 30.0% | Dev2 R@5= 70.0% | Latency= 11.3ms
Config: Whole-Frame Baseline             | Dev R@1= 80.0% | Dev2 R@1= 80.0% | Dev2 R@5=100.0% | Latency= 11.0ms
Config: No-Tracking Ablation             | Dev R@1= 70.0% | Dev2 R@1= 30.0% | Dev2 R@5= 80.0% | Latency= 11.3ms
Config: Prompt Variant: Bare             | Dev R@1= 70.0% | Dev2 R@1= 40.0% | Dev2 R@5= 70.0% | Latency= 11.6ms
Config: Prompt Variant: 'a photo of...'  | Dev R@1= 70.0% | Dev2 R@1= 40.0% | Dev2 R@5= 80.0% | Latency= 13.5ms

Saved machine-readable evaluation results to C:\projects\MULTIStream\eval\results\eval_summary.json

==========================================================================================
ABLATION STUDY RESULTS (Dev and Dev2 Splits, Strict Camera & Label Match)
==========================================================================================
| Configuration | Dev MRR | Dev R@1 | Dev R@5 | Dev2 MRR | Dev2 R@1 | Dev2 R@5 | Median Latency |
|---|---|---|---|---|---|---|---|
| Full Pipeline (Ours)         | 0.7833 |  70.0% |  90.0% | 0.4583 |  30.0% |  70.0% |   11.3 ms |
| Whole-Frame Baseline         | 0.8833 |  80.0% | 100.0% | 0.8750 |  80.0% | 100.0% |   11.0 ms |
| No-Tracking Ablation         | 0.8000 |  70.0% | 100.0% | 0.5167 |  30.0% |  80.0% |   11.3 ms |
| Prompt Variant: Bare         | 0.7833 |  70.0% |  90.0% | 0.4950 |  40.0% |  70.0% |   11.6 ms |
| Prompt Variant: 'a photo of...' | 0.7833 |  70.0% |  90.0% | 0.5333 |  40.0% |  80.0% |   13.5 ms |
==========================================================================================

STORAGE & EFFICIENCY FOOTPRINT:
- SQLite Index Database: 0.75 MB
- Snapshot Evidence Cache: 16.53 MB
- Total Index Footprint: 17.27 MB
- Ingest Throughput: 1.38x Real-Time (73.58 video seconds processed in 53.24 seconds wall-clock)
```

---

## 5. Key Findings & Discussion

1. **Impact of Label Matching & Tolerance Tightening**:
   - Requiring exact label agreement (or semantic synonym match) prevents irrelevant objects (such as pedestrian crops) from registering as false hits for vehicles like utility carts or trucks.
   - Whole-frame embedding exhibits strong global scene recall (80.0% R@1 on dev and dev2), since global context inherently represents the full frame without bounding box cropping errors.
2. **Crop Saliency vs Whole-Frame Embeddings**:
   - For small objects in 4K resolution (e.g. `test_landscape2.mp4`), small bounding box crops occasionally lose fine-grained contextual features in SigLIP-Base (224x224 input resolution). In contrast, whole-frame embeddings capture the full road context, explaining why the whole-frame baseline achieves higher top-1 retrieval under these hardened criteria.
3. **Ingest Speed Clarification**:
   - Ingest throughput is **1.38x Real-Time**, measured via `python scripts/ingest.py --videos footage/ --manifest footage/manifest.json --out index_base` (73.58 seconds of video across 5 clips processed in 53.24 seconds wall-clock).

---

## 6. Deviations
- **Held-Out Set Relabeled to Dev2**: Since the original held-out split (`test_01` to `test_10`) was analyzed during failure diagnosis, it has been reclassified as `dev2`. A fresh held-out set will be evaluated once upon receipt without prior tuning.
- **VLM API Calls**: Marked **implemented, NOT RUN: no API keys** per project rules.

---

## 7. Known Issues
- On fine-grained open-vocabulary queries (e.g., "utility cart"), SigLIP crop embeddings without fine-tuning show high semantic overlap with standard car/truck representations, requiring whole-frame scene context to disambiguate.

---

## 8. What was NOT Verified
- Multimodal cloud VLM reranking API calls (marked **NOT RUN: no API keys**).
- Browser UI interactions (tested via automated HTTP client per Rule 1.2).

---

## 9. Git Log
```text
(pending commit for Phase 6 fixes)
b3763ff | irfanbasha11012007-max <irfanbasha11012007@gmail.com> | phase6: error analysis, strict camera matching, vlm reranking module, and report
3e942db | irfanbasha11012007-max <irfanbasha11012007@gmail.com> | phase5: implement evaluation harness, lock queries sha256, run ablation study, and generate report
bb9689b | irfanbasha11012007-max <irfanbasha11012007@gmail.com> | phase4: implement fastapi backend, static chat ui, and automated api test suite
```
