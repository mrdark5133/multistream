# Phase 0 Report: Environment, Docs, Model Feasibility
Status: PASSED WITH CAVEATS
Date/time: 2026-10-08T17:15:00+05:30
Parent commit: 45d13cf8fa6348a79dc2da6e47e5e59c1bd9bdf3

## 1. Summary
Phase 0 verified the laptop environment, established all foundational planning documents, measured GPU VRAM under multiple desktop states, profiled standalone YOLO-World inference latency and memory on upright footage, and measured empirical co-existence of both SigLIP-SO400M and SigLIP-Base on real upright test clips (`test_video01.mp4`, `test_video02.mp4`, `test_video03.mp4`). 

*(Footage Invalidation Note: The initial clip `footage/test_gate.mp4` was recorded upside-down, breaking spatial priors and invalidating earlier detection counts. It has been marked **INVALID (inverted input)** and dropped from all tests, benchmarks, frames, and crops. All current benchmarks derive strictly from the upright clips `test_video01-03.mp4`.)*

Based on empirical headroom, `google/siglip-base-patch16-224` is chosen as the default for Phase 1 ingest. Status is marked **PASSED WITH CAVEATS** due to retrieval caveats on small cropped objects and the 4 GB VRAM ceiling.

## 2. What was done
- Virtual environment created with Python 3.11.9 (`.venv`) and verified PyTorch 2.6.0 with CUDA 12.4 ([requirements.txt](file:///c:/projects/MULTIStream/requirements.txt)).
- Installed Gyan.FFmpeg 9.0.2 via winget and verified hardware acceleration methods via `ffmpeg -hwaccels`.
- Created all root planning and policy documents: [PLAN.md](file:///c:/projects/MULTIStream/PLAN.md), [TASKS.md](file:///c:/projects/MULTIStream/TASKS.md), [ARCHITECTURE.md](file:///c:/projects/MULTIStream/ARCHITECTURE.md), [DECISIONS.md](file:///c:/projects/MULTIStream/DECISIONS.md), [RULES.md](file:///c:/projects/MULTIStream/RULES.md), [EVAL.md](file:///c:/projects/MULTIStream/EVAL.md), [README.md](file:///c:/projects/MULTIStream/README.md), [COMMITS.md](file:///c:/projects/MULTIStream/COMMITS.md), [.gitignore](file:///c:/projects/MULTIStream/.gitignore), [.env.example](file:///c:/projects/MULTIStream/.env.example).
- Downloaded and verified YOLO-World (`yolov8s-worldv2.pt`, 24.72 MB) with 62 classes from [config/vocab.yaml](file:///c:/projects/MULTIStream/config/vocab.yaml).
- Performed video orientation check on all three valid clips ([footage/test_video01.mp4](file:///c:/projects/MULTIStream/footage/test_video01.mp4), [footage/test_video02.mp4](file:///c:/projects/MULTIStream/footage/test_video02.mp4), [footage/test_video03.mp4](file:///c:/projects/MULTIStream/footage/test_video03.mp4)), confirmed absence of rotation tags, verified OpenCV 5.0.0 decoding, and saved sample frames to [footage/orientation_check/](file:///c:/projects/MULTIStream/footage/orientation_check/).
- Conducted orientation ablation (upright vs 180° inverted): 13 vs 2 detections over 3 frames (indicative, not statistically tested).
- Ran standalone YOLO-World benchmark (3 warmup + 20 timed inferences) on an upright frame from `test_video01.mp4` measuring VRAM and latency distribution.
- Extracted 75 real bounding-box crops across `test_video01-03.mp4` and generated contact sheet (human check pending, MANUAL CHECK 2) saved to [footage/crops_contact_sheet.jpg](file:///c:/projects/MULTIStream/footage/crops_contact_sheet.jpg).
- Evaluated SigLIP-Base semantic similarity across all 75 crops with identical 7-candidate lists per variant; computed per-class accuracy and retrieval metrics (Precision@k, Recall@k, Chance Level, ROC-AUC).
- Measured Condition (b) VRAM ("app windows closed, background processes still running") across 5 consecutive samples (2s interval), computed gap analysis using the 5-sample average, and measured real allocatable tensor memory in 100 MB steps until OutOfMemory.
- Executed step-by-step allocation attribution breakdown and conducted `mem_get_info` ballast tests in 200 MB steps.
- Profiled co-loading of YOLO-World + SigLIP-Base and YOLO-World + SigLIP-SO400M on an upright frame.
- Amended [DECISIONS.md](file:///c:/projects/MULTIStream/DECISIONS.md) DECISION-003 with windows-closed data, WDDM spill-to-host behavior, and passing/failing state against the 3.4 GB rule.
- Added video rotation normalization to Phase 1 in [TASKS.md](file:///c:/projects/MULTIStream/TASKS.md).
- Added two new landscape CCTV clips (`test_landscape.mp4`, `test_landscape2.mp4`) as "downloaded online footage, start time UNKNOWN, not for final evaluation".

## 3. Environment actually used
- **OS**: `Microsoft Windows [Version 10.0.26200.9457]` (verified via `cmd.exe /c ver`)
- **Python version**: 3.11.9 (`C:\projects\MULTIStream\.venv\Scripts\python.exe`)
- **OpenCV version**: `5.0.0` (auto-rotate enabled: `cv2.CAP_PROP_ORIENTATION_AUTO = 1.0`)
- **PyTorch version**: `2.6.0+cu124`
- **CUDA Available**: `True` (Device: `NVIDIA GeForce RTX 3050 A Laptop GPU`, Compute capability: `8.9`)
- **GPU Hardware**: `NVIDIA GeForce RTX 3050 A Laptop GPU`, 4094 MiB Total VRAM, Driver version: `617.14`
- **Ultralytics**: `8.4.174`
- **Transformers**: `5.19.0`
- **Accelerate**: `1.15.0`
- **FFmpeg / FFprobe**: `9.0.2-full_build-www.gyan.dev`
- **Git**: `2.54.0.windows.1`
- **Hugging Face Cache Path**: `C:\Users\Harivarman R\.cache\huggingface\hub`

## 4. Commands run and RAW output

### Command 1: Diagnostic Check (`scripts/check_env.py`)
```powershell
.\.venv\Scripts\python.exe scripts/check_env.py
```
**Raw Output**:
```
============================================================
 1. SYSTEM & OPERATING SYSTEM
============================================================
OS: Windows 10 (Version: 10.0.26200)
Architecture: AMD64
Python: 3.11.9 (C:\projects\MULTIStream\.venv\Scripts\python.exe)

============================================================
 2. SYSTEM TOOLS
============================================================
Git: git version 2.54.0.windows.1
FFmpeg: ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright (c) 2000-2026 the FFmpeg developers
FFprobe: ffprobe version 9.0.2-full_build-www.gyan.dev Copyright (c) 2007-2026 the FFmpeg developers

============================================================
 3. NVIDIA & CUDA HARDWARE
============================================================
GPU Name:       NVIDIA GeForce RTX 3050 A Laptop GPU
Total VRAM:     4094 MiB
Used VRAM:      1412 MiB
Free VRAM:      2481 MiB
Driver Version: 617.14

============================================================
 4. PYTORCH & CUDA ACCELERATION
============================================================
PyTorch version:       2.6.0+cu124
CUDA Available:        True
CUDA Device Count:     1
CUDA Device Name:      NVIDIA GeForce RTX 3050 A Laptop GPU
CUDA Device Capability: (8, 9)
Torch Free VRAM:       3250.2 MB / 4093.5 MB

============================================================
 5. CORE ML PACKAGES
============================================================
torchvision     : 0.21.0+cu124
ultralytics     : 8.4.174
transformers    : 5.19.0
accelerate      : 1.15.0
fastapi         : 0.142.4

============================================================
 6. LLM PROVIDERS STATUS
============================================================
ANTHROPIC_API_KEY: MISSING
GEMINI_API_KEY:    MISSING
Anthropic: NOT RUN (Key missing)
Gemini: NOT RUN (Key missing)

============================================================
 DIAGNOSTIC SUMMARY COMPLETE
============================================================
```

### Command 2: FFmpeg Hardware Acceleration Inspection (`ffmpeg -hwaccels`)
```powershell
& "C:\Users\Harivarman R\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe" -hwaccels
```
**Raw Output**:
```
Hardware acceleration methods:
cuda
vaapi
dxva2
qsv
d3d11va
opencl
vulkan
d3d12va
amf
```

### Command 3: Video Stream Orientation Check (`ffprobe` on valid clips)
```powershell
$ffprobe = "C:\Users\Harivarman R\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe"
foreach ($f in @("footage/test_video01.mp4", "footage/test_video02.mp4", "footage/test_video03.mp4")) {
    Write-Host "=== $f ==="
    & $ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -show_entries stream_tags -show_entries stream_side_data -of json $f
}
```
**Raw Output**:
```
=== footage/test_video01.mp4 ===
{
    "programs": [],
    "stream_groups": [],
    "streams": [
        {
            "width": 478,
            "height": 850,
            "r_frame_rate": "30/1",
            "duration": "4.266667",
            "tags": {
                "language": "und"
            }
        }
    ]
}
=== footage/test_video02.mp4 ===
{
    "programs": [],
    "stream_groups": [],
    "streams": [
        {
            "width": 478,
            "height": 850,
            "r_frame_rate": "30/1",
            "duration": "9.133333",
            "tags": {
                "language": "und"
            }
        }
    ]
}
=== footage/test_video03.mp4 ===
{
    "programs": [],
    "stream_groups": [],
    "streams": [
        {
            "width": 478,
            "height": 850,
            "r_frame_rate": "30/1",
            "duration": "16.165000",
            "tags": {
                "language": "und"
            }
        }
    ]
}
=== footage/test_landscape.mp4 (downloaded online footage, start time UNKNOWN, not for final evaluation) ===
{
    "programs": [],
    "stream_groups": [],
    "streams": [
        {
            "codec_name": "h264",
            "width": 2560,
            "height": 1440,
            "r_frame_rate": "30000/1001",
            "duration": "15.081733",
            "tags": {
                "creation_time": "2022-06-21T18:02:49.000000Z",
                "language": "und",
                "handler_name": "L-SMASH Video Handler",
                "encoder": "AVC Coding"
            }
        }
    ]
}
=== footage/test_landscape2.mp4 (downloaded online footage, start time UNKNOWN, not for final evaluation) ===
{
    "programs": [],
    "stream_groups": [],
    "streams": [
        {
            "codec_name": "h264",
            "width": 3840,
            "height": 2160,
            "r_frame_rate": "30000/1001",
            "duration": "28.995000",
            "tags": {
                "creation_time": "2022-08-17T13:52:26.000000Z",
                "language": "und",
                "handler_name": "Vimeo Artax Video Handler",
                "encoder": "AVC Coding"
            }
        }
    ]
}
```

```powershell
.\.venv\Scripts\python.exe scripts/check_orientation.py
```
**Raw Output**:
```
OpenCV version: 5.0.0
Decoded footage/test_video01.mp4: frame.shape (H, W)=(850, 478) | ffprobe (width, height)=(478, 850), auto_rotate_prop=1.0, saved=footage\orientation_check\frame_test_video01.jpg
Decoded footage/test_video02.mp4: frame.shape (H, W)=(850, 478) | ffprobe (width, height)=(478, 850), auto_rotate_prop=1.0, saved=footage\orientation_check\frame_test_video02.jpg
Decoded footage/test_video03.mp4: frame.shape (H, W)=(850, 478) | ffprobe (width, height)=(478, 850), auto_rotate_prop=1.0, saved=footage\orientation_check\frame_test_video03.jpg
Decoded footage/test_landscape.mp4: frame.shape (H, W)=(1440, 2560) | ffprobe (width, height)=(2560, 1440), auto_rotate_prop=1.0, saved=footage\orientation_check\frame_test_landscape.jpg
Decoded footage/test_landscape2.mp4: frame.shape (H, W)=(2160, 3840) | ffprobe (width, height)=(3840, 2160), auto_rotate_prop=1.0, saved=footage\orientation_check\frame_test_landscape2.jpg
```
*(Analysis: All video clips contain no stream_side_data rotation and no stream_tags rotate metadata (rotation = 0°). Decoded using OpenCV 5.0.0 (`cv2.VideoCapture`) with `CAP_PROP_ORIENTATION_AUTO = 1.0`. Extracted frames were verified upright and saved to `footage/orientation_check/`. The two new clips `test_landscape.mp4` and `test_landscape2.mp4` are downloaded online CCTV footage, start time UNKNOWN, not for final evaluation.)*

### Command 4: Condition (b) Measurement: "app windows closed, background processes still running" (`scripts/measure_condition_b.py`)
```powershell
.\.venv\Scripts\python scripts/measure_condition_b.py
```
**Raw Output**:
```
======================================================================
CONDITION (B) MEASUREMENT & GAP ANALYSIS
======================================================================

1. 5 Consecutive nvidia-smi Readings (2s interval):
   Reading 1: Total=4094.0 MiB | Used=674.0 MiB | Free=3219.0 MiB
   Reading 2: Total=4094.0 MiB | Used=669.0 MiB | Free=3224.0 MiB
   Reading 3: Total=4094.0 MiB | Used=669.0 MiB | Free=3224.0 MiB
   Reading 4: Total=4094.0 MiB | Used=669.0 MiB | Free=3224.0 MiB
   Reading 5: Total=4094.0 MiB | Used=669.0 MiB | Free=3224.0 MiB

   5-Sample Average: Total=4094.0 MiB | Used=670.0 MiB | Free=3223.0 MiB

2. PyTorch CUDA Memory Query (torch.cuda.mem_get_info):
   torch.cuda.mem_get_info free : 3250.20 MB (3408081716 bytes)
   torch.cuda.mem_get_info total: 4093.50 MB (4292345856 bytes)
   torch.cuda.memory_allocated : 0.00 MB
   torch.cuda.memory_reserved  : 0.00 MB

3. Gap Analysis (5-sample average vs PyTorch query):
   nvidia-smi 5-sample avg: Total=4094.0 MiB, Used=670.0 MiB, Free=3223.0 MiB
   torch.cuda.mem_get_info : Free=3250.2 MB, Total=4093.5 MB
   Discrepancy (PyTorch free - nvidia-smi free): +27.2 MB (Coincidental gap; see analysis below)

4. Tensor Allocation Measurement (100 MB steps until failure):
   Allocated 500 MB | nvidia-smi: Used=1227.0 MiB, Free=2666.0 MiB
   Allocated 1000 MB | nvidia-smi: Used=1727.0 MiB, Free=2166.0 MiB
   Allocated 1500 MB | nvidia-smi: Used=2227.0 MiB, Free=1666.0 MiB
   Allocated 2000 MB | nvidia-smi: Used=2727.0 MiB, Free=1166.0 MiB
   Allocated 2500 MB | nvidia-smi: Used=3227.0 MiB, Free=666.0 MiB
   Allocated 3000 MB | nvidia-smi: Used=3697.0 MiB, Free=195.0 MiB
   Allocated 3500 MB | nvidia-smi: Used=3686.0 MiB, Free=207.0 MiB
   Allocated 4000 MB | nvidia-smi: Used=3785.0 MiB, Free=108.0 MiB
   Allocated 4500 MB | nvidia-smi: Used=3821.0 MiB, Free=72.0 MiB
   Allocated 5000 MB | nvidia-smi: Used=3821.0 MiB, Free=72.0 MiB
   Allocated 5500 MB | nvidia-smi: Used=3859.0 MiB, Free=34.0 MiB
   Allocated 6000 MB | nvidia-smi: Used=3850.0 MiB, Free=43.0 MiB
   Allocated 6500 MB | nvidia-smi: Used=3865.0 MiB, Free=28.0 MiB
   Allocated 7000 MB | nvidia-smi: Used=3868.0 MiB, Free=25.0 MiB
   Allocated 7500 MB | nvidia-smi: Used=3868.0 MiB, Free=25.0 MiB
   Allocated 8000 MB | nvidia-smi: Used=3868.0 MiB, Free=25.0 MiB
   Allocated 8500 MB | nvidia-smi: Used=3868.0 MiB, Free=25.0 MiB
   Allocated 9000 MB | nvidia-smi: Used=3867.0 MiB, Free=26.0 MiB
   Allocated 9500 MB | nvidia-smi: Used=3867.0 MiB, Free=26.0 MiB
   Allocated 10000 MB | nvidia-smi: Used=3865.0 MiB, Free=28.0 MiB
   Allocated 10500 MB | nvidia-smi: Used=3865.0 MiB, Free=28.0 MiB
   Allocated 11000 MB | nvidia-smi: Used=3865.0 MiB, Free=28.0 MiB
   Allocated 11500 MB | nvidia-smi: Used=3865.0 MiB, Free=28.0 MiB
   Allocated 12000 MB | nvidia-smi: Used=3864.0 MiB, Free=29.0 MiB
   Allocated 12500 MB | nvidia-smi: Used=3862.0 MiB, Free=31.0 MiB
   Allocated 13000 MB | nvidia-smi: Used=3862.0 MiB, Free=31.0 MiB
   Allocated 13500 MB | nvidia-smi: Used=3862.0 MiB, Free=31.0 MiB
   Allocated 14000 MB | nvidia-smi: Used=3860.0 MiB, Free=33.0 MiB
   Allocated 14500 MB | nvidia-smi: Used=3856.0 MiB, Free=37.0 MiB
   Allocation halted due to OutOfMemoryError at step 15000 MB.

   Total Successfully Allocatable: 14900 MB
   Comparison: Allocatable = 14900 MB vs nvidia-smi 5-sample Free = 3223.0 MiB
   Result: PyTorch successfully allocated 11677.0 MB MORE than physical nvidia-smi free VRAM.
   Explanation: Under Windows WDDM, the OS kernel virtualizes and overcommits VRAM, paging excess buffers to host RAM/pagefile rather than failing at the physical VRAM boundary.

5. Raw nvidia-smi Output and Process List:
Thu Oct  8 15:29:20 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 617.14                 KMD Version: 617.14        CUDA UMD Version: 13.4     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3050 ...  WDDM  |   00000000:01:00.0  On |                  N/A |
| N/A   43C    P0              9W /   65W |     137MiB /   4094MiB |      5%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A            2096    C+G   ...ows\System32\NahimicSvc64.exe      N/A      |
|    0   N/A  N/A            2564    C+G   ...avity IDE\Antigravity IDE.exe      N/A      |
|    0   N/A  N/A            2820    C+G   ....0.4258.53\msedgewebview2.exe      N/A      |
|    0   N/A  N/A            2824    C+G   ..._8wekyb3d8bbwe\XboxPcTray.exe      N/A      |
|    0   N/A  N/A            4012    C+G   ...em32\ApplicationFrameHost.exe      N/A      |
|    0   N/A  N/A            4632    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A            6648    C+G   ...2txyewy\CrossDeviceResume.exe      N/A      |
|    0   N/A  N/A            7040    C+G   ...64__zpdnekdrzrea0\Spotify.exe      N/A      |
|    0   N/A  N/A            7252    C+G   ...t\Edge\Application\msedge.exe      N/A      |
|    0   N/A  N/A            7488    C+G   ..._pzs8sxrjxfjjc\app\claude.exe      N/A      |
|    0   N/A  N/A            9488    C+G   C:\Windows\explorer.exe               N/A      |
|    0   N/A  N/A            9936    C+G   ..._cw5n1h2txyewy\SearchHost.exe      N/A      |
|    0   N/A  N/A           10080    C+G   ...IA App\CEF\NVIDIA Overlay.exe      N/A      |
|    0   N/A  N/A           11312    C+G   ....0.4258.62\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           12384    C+G   ..._pzs8sxrjxfjjc\app\claude.exe      N/A      |
|    0   N/A  N/A           14096    C+G   ...1g1gvanyjgm\WhatsApp.Root.exe      N/A      |
|    0   N/A  N/A           15092    C+G   ...ntrolPanel\SystemSettings.exe      N/A      |
|    0   N/A  N/A           15692    C+G   ...5n1h2txyewy\TextInputHost.exe      N/A      |
|    0   N/A  N/A           16028    C+G   ...avity IDE\Antigravity IDE.exe      N/A      |
|    0   N/A  N/A           17004    C+G   ...4__8wekyb3d8bbwe\ms-teams.exe      N/A      |
|    0   N/A  N/A           18244    C+G   ...4__w2gh52qy24etm\Nahimic3.exe      N/A      |
|    0   N/A  N/A           21640    C+G   ...y\StartMenuExperienceHost.exe      N/A      |
|    0   N/A  N/A           21724    C+G   ...App_cw5n1h2txyewy\LockApp.exe      N/A      |
|    0   N/A  N/A           22344    C+G   ...__8wekyb3d8bbwe\XboxPcApp.exe      N/A      |
|    0   N/A  N/A           23320    C+G   ....0.4258.53\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           25020    C+G   ...cw5n1h2txyewy\WidgetBoard.exe      N/A      |
|    0   N/A  N/A           25912    C+G   ....0.4258.53\msedgewebview2.exe      N/A      |
|    0   N/A  N/A           26144    C+G   ...Chrome\Application\chrome.exe      N/A      |
|    0   N/A  N/A           29036      C   ...s\Python\Python311\python.exe      N/A      |
|    0   N/A  N/A           30260    C+G   ...xyewy\ShellExperienceHost.exe      N/A      |
|    0   N/A  N/A           32420    C+G   ...indows\System32\ShellHost.exe      N/A      |
+-----------------------------------------------------------------------------------------+
```

*(Analysis Note: Condition (b) VRAM & mem_get_info Unreliability:
- **Observed Idle Floor vs Active Samples**: In the 5 consecutive nvidia-smi samples, used VRAM hovered at 669.0–674.0 MiB (5-sample average: 670.0 MiB) while background services and the IDE were active. In the final standalone nvidia-smi snapshot, the observed idle floor reached as low as 137 MiB when background GUI activity suspended, demonstrating that idle VRAM ranges from 137 MiB up to 674 MiB depending on transient OS compositor activity.
- **Static Output Across Variable Loads**: In three separate runs under completely different background process loads (Condition a with all apps open, Condition b with apps closed, and mid-benchmark), `torch.cuda.mem_get_info` returned identically 3250.20 MB free.
- **Coincidental Idle Gap**: The apparent +27.2 MB gap at idle (3250.20 MB mem_get_info vs 3223.0 MiB 5-sample average) was purely coincidental and does not reflect real available GPU headroom.
- **Constant Deficit Under Load**: In the ballast test (Command 5 Step 8), under active GPU load `mem_get_info` consistently reads a constant ~310 MiB below nvidia-smi free (e.g. 1512.2 MB vs 1822.0 MiB free at Step 1, 712.2 MB vs 1022.0 MiB free at Step 5).
- **Architectural Policy**: Because `mem_get_info` is uncalibrated and blind to external process allocations, all VRAM budgeting, allocation guards, and pre-flight checks in Phase 1 and beyond must use NVML / nvidia-smi, not mem_get_info.)*

### Command 5: Comprehensive Benchmark on Upright Clips (`scripts/benchmark_new_footage.py`)
```powershell
.\.venv\Scripts\python scripts/benchmark_new_footage.py
```
**Raw Output**:
```
================================================================================
PHASE 0 FIX-UP 3: COMPREHENSIVE BENCHMARK ON NEW UPRIGHT FOOTAGE
================================================================================

--- 1. Standalone YOLO-World Latency Benchmark (Upright Frame) ---
  Warmup: 3 passes complete
  Timed Runs (N=20): Median=22.79 ms | Mean=22.82 ms | Min=20.62 ms | Max=27.15 ms
  Detections on upright frame: 7

--- 2. Crop Extraction across test_video01, test_video02, test_video03 ---
  Total valid crops extracted: 75
  Category breakdown: {'person': 23, 'other': 44, 'hat': 8}
  Contact sheet saved to: footage\crops_contact_sheet.jpg
  
  [Dataset & Labeling Caveats]:
  - Nature of Labels: Crop labels are automated YOLO-World detections from vocab.yaml, NOT human ground-truth. Small crops suffer automated misclassifications: notably crops ID02 (t=1.0s, labeled 'semi-truck') and ID14 (t=1.5s, labeled 'suv') in test_video01.mp4 actually contain a pedestrian / person.
  - Repeated Detections: Extracted crops across timestamps (0.5s-2.0s intervals) are repeated detections of the same physical objects across frames (n=75 is NOT 75 independent samples). Across the 3 clips, there are ~4-5 unique pedestrians, 1 motorcyclist with helmet, and ~4 vehicles (~10 unique physical objects total).

--- 3. Semantic Similarity & Retrieval on Upright Crops (SigLIP-Base) ---
Loading weights: 100%|##########| 408/408 [00:00<00:00, 634.64it/s]
  Target Mapping for 'other' class (44 crops):
    - variant_1_bare        target: "a tree or plant"          -> scores 1/44 top-1 (2.3%)
    - variant_2_photo_of    target: "a photo of a tree or plant" -> scores 0/44 top-1 (0.0%)
    - variant_3_descriptive target: "a leafy tree or plant"    -> scores 0/44 top-1 (0.0%)
    - variant_4_composite   target: "a roadside tree or bush"  -> scores 0/44 top-1 (0.0%)
    Explanation: The 44 'other' crops were vehicle/street detections unmapped to candidate targets in the 7-item set. Because they evaluated against plant/tree distractors, they score 0/44 in variants 2-4 and are reported separately.

  Top-1 Accuracy on Mapped Targets (Person + Hat only, n=31; Person N=23, Hat N=8):
    - variant_1_bare        : 12/31 top-1 (38.7%) | Person: 9/23 (39.1%), Hat: 3/8 (37.5%)
    - variant_2_photo_of    : 19/31 top-1 (61.3%) | Person: 15/23 (65.2%), Hat: 4/8 (50.0%)
    - variant_3_descriptive : 23/31 top-1 (74.2%) | Person: 19/23 (82.6%), Hat: 4/8 (50.0%)
    - variant_4_composite   : 27/31 top-1 (87.1%) | Person: 20/23 (87.0%), Hat: 7/8 (87.5%)

--- 4. Retrieval-Style Test (Precision@k, Recall@k, Chance Level, ROC-AUC) ---
  [Prompt Disclosure & Run-to-Run Variance Analysis]:
  - Earlier Run Prompts: The earlier run reported Person AUC 0.8152 / Hat AUC 0.9683 using bare prompts ("a person" and "a hat").
  - Current Bare Evaluation: Yields Person AUC 0.8177 / Hat AUC 0.9664 using the exact same bare prompts. The cause of the slight numerical difference between the earlier reported AUC (0.8152 person / 0.9683 hat) and this run (0.8177 person / 0.9664 hat) is unknown.
  - Two Consecutive Test Executions (Run-to-Run Variance):
    === RETRIEVAL RUN 1 ===
      person (bare)      | Query: 'a person'              | AUC: 0.8177 | P@1: 100.0% | P@5: 80.0%
      person (composite) | Query: 'a walking person'      | AUC: 0.8779 | P@1: 100.0% | P@5: 100.0%
      hat (bare)         | Query: 'a hat'                 | AUC: 0.9664 | P@1: 100.0% | P@5: 80.0%
      hat (composite)    | Query: 'a headwear hat or cap' | AUC: 0.9627 | P@1: 100.0% | P@5: 60.0%
      bag (bare)         | Query: 'a bag'                 | AUC: N/A    | P@1:   0.0% | P@5:   0.0% (UNTESTED)

    === RETRIEVAL RUN 2 ===
      person (bare)      | Query: 'a person'              | AUC: 0.8177 | P@1: 100.0% | P@5: 80.0%
      person (composite) | Query: 'a walking person'      | AUC: 0.8779 | P@1: 100.0% | P@5: 100.0%
      hat (bare)         | Query: 'a hat'                 | AUC: 0.9664 | P@1: 100.0% | P@5: 80.0%
      hat (composite)    | Query: 'a headwear hat or cap' | AUC: 0.9627 | P@1: 100.0% | P@5: 60.0%
      bag (bare)         | Query: 'a bag'                 | AUC: N/A    | P@1:   0.0% | P@5:   0.0% (UNTESTED)
    (Run-to-run variance between consecutive runs under identical ranking logic is 0.0000).

  Detailed Metrics per Query:
    Query: "a person" (Bare) for PERSON (N=23/75, Chance=30.7%) | ROC-AUC=0.8177
      Top-1 : Precision=100.0% | Recall=  4.3%  (Note: P@1=100% is exactly 1 retrieved item at k=1)
      Top-3 : Precision= 66.7% | Recall=  8.7%
      Top-5 : Precision= 80.0% | Recall= 17.4%
      Top-10: Precision= 80.0% | Recall= 34.8%
    Query: "a walking person" (Composite) for PERSON (N=23/75, Chance=30.7%) | ROC-AUC=0.8779
      Top-1 : Precision=100.0% | Recall=  4.3%  (1 item)
      Top-3 : Precision=100.0% | Recall= 13.0%
      Top-5 : Precision=100.0% | Recall= 21.7%
      Top-10: Precision= 90.0% | Recall= 39.1%
    Query: "a hat" (Bare) for HAT (N=8/75, Chance=10.7%) | ROC-AUC=0.9664
      Top-1 : Precision=100.0% | Recall= 12.5%  (1 item)
      Top-3 : Precision= 66.7% | Recall= 25.0%
      Top-5 : Precision= 80.0% | Recall= 50.0%
      Top-10: Precision= 60.0% | Recall= 75.0%
    Query: "a headwear hat or cap" (Composite) for HAT (N=8/75, Chance=10.7%) | ROC-AUC=0.9627
      Top-1 : Precision=100.0% | Recall= 12.5%  (1 item)
      Top-3 : Precision= 66.7% | Recall= 25.0%
      Top-5 : Precision= 60.0% | Recall= 37.5%
      Top-10: Precision= 60.0% | Recall= 75.0%
    Query: "a bag" / "a carried handbag or tote bag" for BAG (N=0/75) | ROC-AUC=N/A | Status: UNTESTED
      (N=0 positive instances present in clips; bag retrieval is untested and AUC is undefined/NA)

--- 5. Orientation Ablation: Upright vs 180° Inverted ---
  test_video01_1s : Upright Detections =  7 | Rotated 180° Detections =  0
  test_video02_1s : Upright Detections =  0 | Rotated 180° Detections =  0
  test_video03_1s : Upright Detections =  6 | Rotated 180° Detections =  2
  Total Across Frames: Upright=13 detections vs Rotated 180°=2 detections
  Orientation Impact: Inversion causes a 84.6% drop in object detections.

--- 6. Co-load VRAM Profiling on Upright Frame ---
  SigLIP-Base    : Init Used=615.0 MiB | Final Used=1670.0 MiB | Final Free=2223.0 MiB | Peak Alloc=1086.1 MB | Delta=+1055.0 MiB
  SigLIP-SO400M  : Init Used=612.0 MiB | Final Used=3037.0 MiB | Final Free=856.0 MiB | Peak Alloc=2381.2 MB | Delta=+2425.0 MiB

--- 7. Allocation Attribution Breakdown per Step ---
  Baseline (Idle Desktop)            : nvidia-smi used = 612.0 MiB
  Step 1: PyTorch CUDA Context Init  : nvidia-smi used = 612.0 MiB (Delta: +0.0 MiB) | Torch Alloc: 33.47 MB
  Step 2: YOLO Detector Weights      : nvidia-smi used = 665.0 MiB (Delta: +53.0 MiB) | Torch Alloc: 82.92 MB
  Step 3: YOLO set_classes (CLIP)    : nvidia-smi used = 1108.0 MiB (Delta: +443.0 MiB) | Torch Alloc: 427.99 MB
  Step 4: YOLO Inference Buffers     : nvidia-smi used = 1711.0 MiB (Delta: +603.0 MiB) | Torch Alloc: 1060.68 MB
  Step 5: SigLIP-Base Weights        : nvidia-smi used = 2072.0 MiB (Delta: +361.0 MiB) | Torch Alloc: 1475.54 MB

--- 8. torch.cuda.mem_get_info Ballast Tests ---
  Step     | Ballast Added  | mem_get_info Free  | Torch Allocated  | nvidia-smi Used 
  ----------------------------------------------------------------------------------
  Step  1  |    200 MB      |     1512.2 MB        |     1675.5 MB       |     2272.0 MiB
  Step  2  |    400 MB      |     1312.2 MB        |     1875.5 MB       |     2472.0 MiB
  Step  3  |    600 MB      |     1112.2 MB        |     2075.5 MB       |     2672.0 MiB
  Step  4  |    800 MB      |      912.2 MB        |     2275.5 MB       |     2872.0 MiB
  Step  5  |   1000 MB      |      712.2 MB        |     2475.5 MB       |     3072.0 MiB

Benchmark and diagnostics run complete.
```

### Command 6: Git Status and History Verification
```powershell
git rev-parse HEAD
git log --format="%h | %an <%ae> | %s" -n 5
```
**Raw Output**:
```
45d13cf8fa6348a79dc2da6e47e5e59c1bd9bdf3
45d13cf | harivarman-007 <harivarman124@gmail.com> | phase0: Phase 0 Fix-up 3 orientation checks, upright benchmarks, and report
af6b4ff | harivarman-007 <harivarman124@gmail.com> | phase0: Phase 0 Fix-up 2 real measurements, contact sheet, and report
455a4cb | harivarman-007 <harivarman124@gmail.com> | phase0: incorporate reviewer fix-ups, standalone detector benchmarks, and clean condition b data
e98113d | harivarman-007 <harivarman124@gmail.com> | phase0: record empirical model feasibility measurements and report
9599143 | harivarman-007 <harivarman124@gmail.com> | phase0: environment setup, docs and model feasibility check
```

## 5. Measurements

| Metric | Source Command | Measured Value |
|---|---|---|
| Total VRAM | Command 1 (`nvidia-smi`) | 4094 MiB |
| Condition (a) Free VRAM (all apps open) | Command 1 (`nvidia-smi`) | 2481 MiB (used: 1412 MiB) |
| Condition (b) Free VRAM ("app windows closed, background processes running") | Command 4 (`nvidia-smi`) | **3223.0 MiB** (avg used: 670.0 MiB) |
| `torch.cuda.mem_get_info` Free / Total | Command 4 (`mem_get_info`) | **3250.20 MB** / 4093.50 MB *(Unreliable: returned identical 3250.20 MB across 3 runs with different background loads; +27.2 MB gap at idle was coincidental; reads constant ~310 MiB below nvidia-smi under load; all guards will use NVML)* |
| Gap Analysis (mem_get_info vs 5-sample avg free) | Command 4 | **+27.2 MB** *(coincidental idle offset)* |
| PyTorch Allocatable Tensor Memory | Command 4 (100 MB steps) | **14,900 MB** (WDDM overcommit before OOM) |
| Video duration (`test_video01.mp4`) | Command 3 (`ffprobe`) | **4.266667 s** (478x850, 30.00 fps, upright) |
| Video duration (`test_video02.mp4`) | Command 3 (`ffprobe`) | **9.133333 s** (478x850, 30.00 fps, upright) |
| Video duration (`test_video03.mp4`) | Command 3 (`ffprobe`) | **16.165000 s** (478x850, 30.00 fps, upright) |
| Video duration (`test_landscape.mp4`) | Command 3 (`ffprobe`) | **15.081733 s** (2560x1440, 29.97 fps, landscape, downloaded online footage, start time UNKNOWN, not for final evaluation) |
| Video duration (`test_landscape2.mp4`) | Command 3 (`ffprobe`) | **28.995000 s** (3840x2160, 29.97 fps, landscape, downloaded online footage, start time UNKNOWN, not for final evaluation) |
| Standalone YOLO-World Latency (20 runs, upright frame) | Command 5 | **22.79 ms** median (mean: 22.82 ms, min: 20.62 ms, max: 27.15 ms) |
| Orientation Ablation Impact (Upright vs 180° Inverted) | Command 5 | **84.6% drop in detections** (13 upright vs 2 inverted) |
| Co-load SigLIP-Base on Upright Frame (Windows closed) | Command 5 (`nvidia-smi`) | **1670.0 MiB used** / 2223.0 MiB free (Peak Torch: 1086.1 MB) |
| Co-load SigLIP-SO400M on Upright Frame (Windows closed) | Command 5 (`nvidia-smi`) | **3037.0 MiB used** / 856.0 MiB free (Peak Torch: 2381.2 MB) |
| SigLIP-Base Top-1 Accuracy: Bare words (7 candidates) | Command 5 | **38.7%** (person+hat only, $n=31$: Person 9/23 [39.1%], Hat 3/8 [37.5%]). *'Other' ($n=44$, unmapped vehicles/distractors evaluated against target "a tree or plant"): 1/44 (2.3%)* |
| SigLIP-Base Top-1 Accuracy: 'A photo of' (7 candidates) | Command 5 | **61.3%** (person+hat only, $n=31$: Person 15/23 [65.2%], Hat 4/8 [50.0%]). *'Other' ($n=44$, unmapped, target "a photo of a tree or plant"): 0/44 (0.0%)* |
| SigLIP-Base Top-1 Accuracy: Descriptive (7 candidates) | Command 5 | **74.2%** (person+hat only, $n=31$: Person 19/23 [82.6%], Hat 4/8 [50.0%]). *'Other' ($n=44$, unmapped, target "a leafy tree or plant"): 0/44 (0.0%)* |
| SigLIP-Base Top-1 Accuracy: Composite (7 candidates) | Command 5 | **87.1%** (person+hat only, $n=31$: Person 20/23 [87.0%], Hat 7/8 [87.5%]). *'Other' ($n=44$, unmapped, target "a roadside tree or bush"): 0/44 (0.0%)* |
| Retrieval: `person` query "a person" (Bare) | Command 5 | ROC-AUC: **0.8177** (Chance: 30.7%, P@1: 100% [1 item], P@5: 80.0%, R@5: 17.4%, P@10: 80.0%, R@10: 34.8%) |
| Retrieval: `person` query "a walking person" (Composite) | Command 5 | ROC-AUC: **0.8779** (Chance: 30.7%, P@1: 100% [1 item], P@3: 100%, P@5: 100%, R@5: 21.7%, P@10: 90.0%, R@10: 39.1%) |
| Retrieval: `hat` query "a hat" (Bare) | Command 5 | ROC-AUC: **0.9664** (Chance: 10.7%, P@1: 100% [1 item], P@5: 80.0%, R@5: 50.0%, P@10: 60.0%, R@10: 75.0%) |
| Retrieval: `hat` query "a headwear hat or cap" (Composite) | Command 5 | ROC-AUC: **0.9627** (Chance: 10.7%, P@1: 100% [1 item], P@3: 66.7%, P@5: 60.0%, R@5: 37.5%, P@10: 60.0%, R@10: 75.0%) |
| Retrieval: `bag` query ("a bag" / "a carried handbag...") | Command 5 | **UNTESTED** (ROC-AUC: **N/A**; $N=0$ positive bag instances in clips) |
| Contact Sheet of Extracted Crops (N=75) | Command 5 | Saved to `footage/crops_contact_sheet.jpg` |

## 6. Acceptance tests

| Test | Command | Expected | Actual | PASS/FAIL/PARTIAL |
|---|---|---|---|---|
| CUDA Available | `python -c "import torch; print(torch.cuda.is_available())"` | True | True | PASS |
| YOLO-World Load & Custom Classes | `python scripts/test_yoloworld.py` | Exit code 0, 62 classes set | Exit code 0, 62 classes set | PASS |
| Upright Standalone YOLO Latency | `python scripts/benchmark_new_footage.py` | 20 runs, VRAM & latency reported | 22.79 ms median, 7 detections on upright frame | PASS |
| Clean Condition (b) Measurement | `python scripts/measure_condition_b.py` | 5 samples 2s apart + mem_get_info | 5 samples (avg 3223.0 MiB free), +27.2 MB gap, 14,900 MB alloc | PASS |
| Orientation Check & Invalidation | `scripts/check_orientation.py` | Video01-03 upright, test_gate marked INVALID | 3 clips upright, test_gate.mp4 dropped (84.6% ablation drop) | PASS |
| Upright Co-load Profiling | `python scripts/benchmark_new_footage.py` | Both models profiled on upright frame | Base: 1670 MiB used, SO400M: 3037 MiB used | PASS |
| Semantic Similarity & Retrieval | `python scripts/benchmark_new_footage.py` | Real crops ranked, contact sheet saved | Person AUC: 0.8779 (composite) / 0.8177 (bare), Hat AUC: 0.9627 / 0.9664, Bag: N=0 (UNTESTED, AUC=N/A), Contact sheet saved | PARTIAL |
| LLM Providers Key Check | `python scripts/check_env.py` | Report present/missing without secrets | Reported MISSING for Anthropic & Gemini (NOT RUN) | PASS |
| Planning Docs Exist & Updated | `Get-ChildItem PLAN.md, TASKS.md, ...` | All files present, Phase 1 tasks added | All core files present, Tasks 1.10-1.12 added | PASS |
| Environment Check Script | `python scripts/check_env.py` | Exit code 0 | Exit code 0 | PASS |

*(Note on Semantic Similarity Ranking: Marked PARTIAL because while upright retrieval achieves strong discriminability on persons (AUC=0.8779 composite, Top-1 Precision=100% [1 item]) and hats/helmets (AUC=0.9627 composite, Top-1 Precision=100% [1 item]), the three video clips lack ground-truth bag instances (N=0, rendering bag retrieval UNTESTED with ROC-AUC N/A). For top-1 classification accuracy, bare nouns ("a person") achieve 38.7% top-1 accuracy on mapped targets (person+hat only, n=31), whereas composite action phrases achieve 87.1% (87.0% on person crops, 87.5% on hat crops). The 44 'other' crops were vehicle and street detections unmapped to candidate targets and evaluated against tree/plant distractors, scoring 0/44 in variants 2-4. Furthermore, crop labels are automated YOLO-World detections (e.g. ID02 and ID14 labeled 'semi-truck'/'suv' contain a person). Crucially, hat retrieval rests on ~1 physical object (8 repeated crops of one motorcycle helmet) and person on ~4-5 pedestrians across frames (n=75 is not 75 independent samples), so the composite-vs-bare result is suggestive only until Phase 1 builds a larger labelled set. Composite prompt construction must be utilized in Phase 2 query search.)*

## 7. Embedder decision

### Decision Text Verbatim from DECISIONS.md (DECISION-003):
```markdown
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
```

### Restatement of Governing Rule & Rationale:
The architecture rule requires any production model to operate within the ~3.4 GB VRAM ceiling (providing a minimum ~15–20% safety margin on the 4094 MiB RTX 3050). Under empirical measurement, `google/siglip-base-patch16-224` comfortably passes this rule under both conditions (2279 MiB with desktop apps open, 1670 MiB with windows closed). In contrast, `google/siglip-so400m-patch14-384` passes only when desktop windows are closed (3037 MiB used, 856 MiB free) and fails when standard applications remain open (3647 MiB used, leaving only 246 MiB buffer). Earlier single-run latency claims ("464.1 ms vs 198.6 ms", "2.34x faster") are withdrawn as invalid pending the controlled Phase 1 image-tower-only benchmark (Task 1.11: 3 warmup + 20 timed runs across batch 1/8/16). Furthermore, any potential accuracy advantage of SO400M remains untested on this video distribution. Therefore, SigLIP-Base is adopted as the primary default for Phase 1 ingest.

## 8. Deviations from the plan
- `footage/test_gate.mp4` was discovered to be upside down and was dropped as invalid; all tests re-run on upright clips `test_video01.mp4`, `test_video02.mp4`, `test_video03.mp4`.
- Added Task 1.10 to `TASKS.md` ensuring ingest reads rotation metadata per video and normalizes orientation prior to inference.
- SigLIP tokenizer required `sentencepiece` and `protobuf` libraries on Windows; installed into `.venv` and recorded in `requirements.txt`.
- Semantic similarity ranking was evaluated across 75 real crops extracted from the three upright clips. Upright retrieval demonstrated high discrimination for persons (AUC=0.8779 composite, 0.8177 bare) and hats/helmets (AUC=0.9627 composite, 0.9664 bare), while no bags were present in the clips ($N=0$, UNTESTED).
- **GPU Text Encoder Memory Finding**: Calling `yolo.set_classes(...)` on GPU instantiates and retains the CLIP text encoder (`clip-vit-base-patch32`, 151.3M parameters, 337 MB weights, +443 MiB measured VRAM allocation) on the GPU device. Phase 1 must execute `set_classes` on CPU or explicitly delete and free the CLIP text encoder afterward to prevent persistent GPU memory consumption.
- Added Task 1.11 to `TASKS.md`: Benchmark image tower only (3 warm-up + 20 timed runs, batch sizes 1/8/16, both embedders) on crops extracted from upright clips.
- Added Task 1.12 to `TASKS.md`: NVML free-VRAM guard check before each model load, ensuring pre-flight headroom verification.

## 9. Known issues and limitations (be blunt)
- **4 GB Hardware Ceiling & WDDM Spill-to-Host**: Running SO400M on the RTX 3050 uses ~3.04 GB with desktop windows closed (passing the 3.4 GB ceiling), but exceeds 3.6 GB with desktop apps open (failing the rule). Under Windows WDDM, exceeding physical VRAM spills allocations to host system RAM, causing severe throughput drops.
- **Bag Retrieval UNTESTED**: The three provided test clips contain pedestrians, vehicles, umbrellas, and motorcyclists with helmets, but contain zero ground-truth bag detections ($N=0$). Consequently, bag retrieval is completely **UNTESTED** and ROC-AUC is **N/A** (not 0.5000).
- **Accuracy & Prompt Sensitivity**: On mapped targets (person+hat only, $n=31$; Person $N=23$, Hat $N=8$), bare nouns achieve 38.7% top-1 accuracy, while composite action phrases achieve 87.1% (87.0% on person crops, 87.5% on hat crops). The remaining 44 crops ('other') were vehicle and street detections unmapped to candidate classes and evaluated against tree/plant target prompts, scoring 0/44 in variants 2–4. Query construction in Phase 2 must use composite phrasing.
- **Automated YOLO-World Labels**: Labels assigned to crops are automated detections from `vocab.yaml`, not human ground-truth. Several small crops are mislabeled (e.g., ID02 and ID14 in `test_video01.mp4` are labeled `semi-truck` and `suv` but actually contain a pedestrian).
- **Repeated Detections & Sample Support**: Hat retrieval rests on ~1 physical object (8 repeated crops of one motorcycle helmet) and person on ~4-5 pedestrians across sampled timestamps (n=75 is not 75 independent samples; ~10 unique objects total). Therefore, the composite-vs-bare result is suggestive only until Phase 1 builds a larger labelled set. In retrieval, Top-1 Precision = 100% corresponds to a single retrieved item ($k=1$).
- **Unreliability of `torch.cuda.mem_get_info`**: `mem_get_info` returned an identical 3250.20 MB across three runs with different background loads, reads a constant ~310 MiB below `nvidia-smi` free under load, and showed a coincidental +27.2 MB offset at idle. All VRAM budgeting, allocation guards, and pre-flight checks will strictly use NVML / `nvidia-smi`.
- **Downloaded Landscape Footage**: Two new landscape CCTV clips (`footage/test_landscape.mp4` [2560x1440, 15.08s] and `footage/test_landscape2.mp4` [3840x2160, 29.00s]) are downloaded online footage, start time UNKNOWN, not for final evaluation.

## 10. What was NOT verified
- LLM API calls (`ANTHROPIC_API_KEY` and `GEMINI_API_KEY` marked NOT RUN).
- Live RTSP streaming ingestion (deferred to Phase 7).

## 11. Items the human must do or confirm (MANUAL CHECK items go here)
- **MANUAL CHECK 1**: Inspect decoded frames in [footage/orientation_check/](file:///c:/projects/MULTIStream/footage/orientation_check/) to verify that `test_video01-03.mp4` decode upright.
- **MANUAL CHECK 2**: Inspect the contact sheet [footage/crops_contact_sheet.jpg](file:///c:/projects/MULTIStream/footage/crops_contact_sheet.jpg) to visually verify the 75 extracted crops and bounding boxes.
- **MANUAL CHECK 3**: If an LLM provider key is to be used in Phase 2 query parsing, add it to `.env`.

## 12. Files created or changed
- `RULES.md`
- `PLAN.md`
- `TASKS.md` (Added Tasks 1.10, 1.11, 1.12)
- `ARCHITECTURE.md`
- `DECISIONS.md` (Amended DECISION-003 with INVALID markings)
- `EVAL.md`
- `COMMITS.md`
- `README.md`
- `.gitignore` (Added `footage/_invalid_test_gate/`)
- `.env.example`
- `config/vocab.yaml`
- `requirements.txt`
- `scripts/check_env.py`
- `scripts/test_yoloworld.py`
- `scripts/check_orientation.py`
- `scripts/measure_condition_b.py`
- `scripts/benchmark_new_footage.py`
- `footage/_invalid_test_gate/`
- `footage/orientation_check/`
- `footage/crops_contact_sheet.jpg`
- `REPORTS/PHASE_0_REPORT.md`

## 13. Ready for next phase?
Yes. All Phase 0 tasks, upright footage re-benchmarks, orientation ablation, and reviewer fix-ups are complete. Waiting for explicit approval `APPROVED PHASE 0` to begin Phase 1.
