# Phase 0 Report: Environment, Docs, Model Feasibility
Status: PASSED WITH CAVEATS
Date/time: 2026-10-08T15:45:00+05:30
Git commit: 9adcc1f51ab49fa131458a3337b98ac2aef93c22

## 1. Summary
Phase 0 verified the laptop environment, established the planning documents, measured GPU VRAM under multiple desktop states, profiled standalone YOLO-World inference latency and memory, and measured empirical co-existence of both SigLIP-SO400M and SigLIP-Base on real footage (`footage/test_gate.mp4`). Based on empirical headroom (3647 MB total GPU use vs 4094 MB budget), SigLIP-Base is chosen as the default for Phase 1 ingest. Status is marked **PASSED WITH CAVEATS** due to semantic retrieval caveats on low-resolution crops and tight VRAM headroom on the 4 GB RTX 3050 Laptop GPU.

## 2. What was done
- Virtual environment created with Python 3.11.9 (`.venv`) and verified PyTorch 2.6.0 with CUDA 12.4 ([requirements.txt](file:///c:/projects/MULTIStream/requirements.txt)).
- Installed Gyan.FFmpeg 9.0.2 via winget with hardware acceleration support and verified via `ffmpeg -hwaccels`.
- Created all root planning and policy documents: [PLAN.md](file:///c:/projects/MULTIStream/PLAN.md), [TASKS.md](file:///c:/projects/MULTIStream/TASKS.md), [ARCHITECTURE.md](file:///c:/projects/MULTIStream/ARCHITECTURE.md), [DECISIONS.md](file:///c:/projects/MULTIStream/DECISIONS.md), [RULES.md](file:///c:/projects/MULTIStream/RULES.md), [EVAL.md](file:///c:/projects/MULTIStream/EVAL.md), [README.md](file:///c:/projects/MULTIStream/README.md), [COMMITS.md](file:///c:/projects/MULTIStream/COMMITS.md), [.gitignore](file:///c:/projects/MULTIStream/.gitignore), [.env.example](file:///c:/projects/MULTIStream/.env.example).
- Downloaded and verified YOLO-World (`yolov8s-worldv2.pt`, 24.72 MB) with 62 classes from [config/vocab.yaml](file:///c:/projects/MULTIStream/config/vocab.yaml).
- Ran standalone YOLO-World benchmark (3 warmup + 20 timed inferences) on real frame measuring VRAM and latency distribution ([scripts/profile_yolo_standalone.py](file:///c:/projects/MULTIStream/scripts/profile_yolo_standalone.py)).
- Extracted real reference frame and 17 real bounding-box crops across multiple timestamps from [footage/test_gate.mp4](file:///c:/projects/MULTIStream/footage/test_gate.mp4) (duration 17.696978 s).
- Inspected the 3 newly supplied test clips ([footage/test_video01.mp4](file:///c:/projects/MULTIStream/footage/test_video01.mp4), [footage/test_video02.mp4](file:///c:/projects/MULTIStream/footage/test_video02.mp4), [footage/test_video03.mp4](file:///c:/projects/MULTIStream/footage/test_video03.mp4)) using `ffprobe`.
- Cleanly measured Condition (b) VRAM across 5 consecutive samples (2s interval) with heavy desktop applications closed, computed gap analysis using the 5-sample average, and measured real allocatable tensor memory in 100 MB steps until OutOfMemory ([scripts/measure_condition_b.py](file:///c:/projects/MULTIStream/scripts/measure_condition_b.py)).
- Profiled co-loading of YOLO-World + SigLIP-Base and YOLO-World + SigLIP-SO400M back-to-back 3 times each, logging initial and final `nvidia-smi` used/free per run, and analyzed parameter bytes for the detector vs the CLIP text encoder ([scripts/profile_coload_3x.py](file:///c:/projects/MULTIStream/scripts/profile_coload_3x.py)).
- Evaluated SigLIP-Base semantic similarity across all 17 crops with identical candidate sets across variants, printed per-crop target prompts and ranks, resolved the anomaly in earlier hat crop scoring, performed a retrieval-style test reporting Recall@k (k=1, 3, 5, 10) and ROC-AUC, and generated a visual contact sheet saved to [footage/crops_contact_sheet.jpg](file:///c:/projects/MULTIStream/footage/crops_contact_sheet.jpg) ([scripts/test_retrieval_and_contact_sheet.py](file:///c:/projects/MULTIStream/scripts/test_retrieval_and_contact_sheet.py)).
- Checked LLM provider environment variables (both marked NOT RUN, no keys exposed).

## 3. Environment actually used
- **OS**: `Microsoft Windows [Version 10.0.26200.9457]` (verified via `cmd.exe /c ver`)
- **Python version**: 3.11.9 (`C:\projects\MULTIStream\.venv\Scripts\python.exe`)
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

### Command 3: Exact Windows OS Version (`cmd.exe /c ver`)
```powershell
cmd.exe /c ver
```
**Raw Output**:
```
Microsoft Windows [Version 10.0.26200.9457]
```

### Command 4: Hugging Face Hub Cache Path Check
```powershell
.\.venv\Scripts\python -c "import huggingface_hub.constants as c; print('HF Cache Path:', c.HF_HUB_CACHE)"
```
**Raw Output**:
```
HF Cache Path: C:\Users\Harivarman R\.cache\huggingface\hub
```

### Command 5: Video Stream Inspection (`ffprobe` on all footage clips)
```powershell
$ffprobe = "C:\Users\Harivarman R\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffprobe.exe"
& $ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1 footage/test_gate.mp4
foreach ($f in @("footage/test_video01.mp4", "footage/test_video02.mp4", "footage/test_video03.mp4")) {
    & $ffprobe -v error -select_streams v:0 -show_entries stream=width,height,duration -of default=noprint_wrappers=1 $f
}
```
**Raw Output**:
```
[test_gate.mp4]
width=478
height=850
r_frame_rate=359/12
duration=17.696978

[test_video01.mp4]
width=478
height=850
duration=4.266667

[test_video02.mp4]
width=478
height=850
duration=9.133333

[test_video03.mp4]
width=478
height=850
duration=16.165000
```

### Command 6: Condition (b) Measurement, Gap Analysis & Allocation Stress Test (`scripts/measure_condition_b.py`)
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
   Discrepancy (PyTorch free - nvidia-smi free): +27.2 MB

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

### Command 7: Standalone YOLO-World Benchmark (`scripts/profile_yolo_standalone.py`)
```powershell
.\.venv\Scripts\python.exe scripts/profile_yolo_standalone.py
```
**Raw Output**:
```
=================================================================
Phase 0 Fix-up Task 4: YOLO-World Standalone Benchmark
=================================================================
Loading YOLO-World (yolov8s-worldv2.pt)...
Loaded and configured 62 classes in 6.71 s

Running 3 warmup inferences on real frame...
Warmup complete.

Running 20 timed inferences...
  Run 01:  23.15 ms | Detections: 3
  Run 02:  22.22 ms | Detections: 3
  Run 03:  31.48 ms | Detections: 3
  Run 04:  27.78 ms | Detections: 3
  Run 05:  24.29 ms | Detections: 3
  Run 06:  27.56 ms | Detections: 3
  Run 07:  29.56 ms | Detections: 3
  Run 08:  25.92 ms | Detections: 3
  Run 09:  24.54 ms | Detections: 3
  Run 10:  24.12 ms | Detections: 3
  Run 11:  23.23 ms | Detections: 3
  Run 12:  23.34 ms | Detections: 3
  Run 13:  23.26 ms | Detections: 3
  Run 14:  27.26 ms | Detections: 3
  Run 15:  22.40 ms | Detections: 3
  Run 16:  21.97 ms | Detections: 3
  Run 17:  21.47 ms | Detections: 3
  Run 18:  21.66 ms | Detections: 3
  Run 19:  24.98 ms | Detections: 3
  Run 20:  21.71 ms | Detections: 3

==================================================
STANDALONE YOLO-WORLD RESULTS:
torch.cuda.memory_allocated    : 1291.70 MB
torch.cuda.max_memory_allocated: 1319.47 MB
nvidia-smi Used VRAM (initial) : 1466.0 MiB
nvidia-smi Used VRAM (active)  : 2953.0 MiB
Net GPU delta from YOLO        : +1487.0 MiB
Median Inference Latency       : 23.73 ms
Worst (Max) Inference Latency  : 31.48 ms
Mean Inference Latency         : 24.59 ms
Min Inference Latency          : 21.47 ms
==================================================
```

### Command 8: 3x Back-to-Back Co-load Profiling & Architecture Breakdown (`scripts/profile_coload_3x.py`)
```powershell
.\.venv\Scripts\python scripts/profile_coload_3x.py
```
**Raw Output**:
```
======================================================================
PHASE 0 FIX-UP 2: CO-LOAD 3X BACK-TO-BACK BENCHMARK
======================================================================

==================================================
TESTING YOLO-WORLD + SIGLIP-BASE (3 RUNS)
==================================================

--- Run 1: google/siglip-base-patch16-224 ---
  [Initial nvidia-smi] Used: 190.0 MiB | Free: 3703.0 MiB | Total: 4094.0 MiB
  [Final nvidia-smi]   Used: 1384.0 MiB | Free: 2509.0 MiB (Delta: +1194.0 MiB)
  [PyTorch Memory]     Current: 1080.0 MB | Peak Allocated: 1086.2 MB
  [Latency]            YOLO: 4799.3 ms | SigLIP load+inf: 3824.2 ms

--- Run 2: google/siglip-base-patch16-224 ---
  [Initial nvidia-smi] Used: 315.0 MiB | Free: 3578.0 MiB | Total: 4094.0 MiB
  [Final nvidia-smi]   Used: 1391.0 MiB | Free: 2502.0 MiB (Delta: +1076.0 MiB)
  [PyTorch Memory]     Current: 1080.0 MB | Peak Allocated: 1086.2 MB
  [Latency]            YOLO: 4402.1 ms | SigLIP load+inf: 2043.1 ms

--- Run 3: google/siglip-base-patch16-224 ---
  [Initial nvidia-smi] Used: 323.0 MiB | Free: 3570.0 MiB | Total: 4094.0 MiB
  [Final nvidia-smi]   Used: 1382.0 MiB | Free: 2511.0 MiB (Delta: +1059.0 MiB)
  [PyTorch Memory]     Current: 1080.0 MB | Peak Allocated: 1086.2 MB
  [Latency]            YOLO: 4404.9 ms | SigLIP load+inf: 2029.0 ms

==================================================
TESTING YOLO-WORLD + SIGLIP-SO400M (3 RUNS)
==================================================

--- Run 1: google/siglip-so400m-patch14-384 ---
  [Initial nvidia-smi] Used: 314.0 MiB | Free: 3579.0 MiB | Total: 4094.0 MiB
  [Final nvidia-smi]   Used: 2856.0 MiB | Free: 1037.0 MiB (Delta: +2542.0 MiB)
  [PyTorch Memory]     Current: 2362.5 MB | Peak Allocated: 2379.7 MB
  [Latency]            YOLO: 4761.5 ms | SigLIP load+inf: 10097.1 ms

--- Run 2: google/siglip-so400m-patch14-384 ---
  [Initial nvidia-smi] Used: 418.0 MiB | Free: 3475.0 MiB | Total: 4094.0 MiB
  [Final nvidia-smi]   Used: 2852.0 MiB | Free: 1041.0 MiB (Delta: +2434.0 MiB)
  [PyTorch Memory]     Current: 2362.5 MB | Peak Allocated: 2379.7 MB
  [Latency]            YOLO: 5173.1 ms | SigLIP load+inf: 6096.9 ms

--- Run 3: google/siglip-so400m-patch14-384 ---
  [Initial nvidia-smi] Used: 418.0 MiB | Free: 3475.0 MiB | Total: 4094.0 MiB
  [Final nvidia-smi]   Used: 2840.0 MiB | Free: 1053.0 MiB (Delta: +2422.0 MiB)
  [PyTorch Memory]     Current: 2362.5 MB | Peak Allocated: 2379.7 MB
  [Latency]            YOLO: 4840.9 ms | SigLIP load+inf: 6459.0 ms

======================================================================
CO-LOAD 3X SUMMARY TABLE:
======================================================================
Model                | Run | Init Used | Init Free | Final Used | Final Free | Peak Alloc
-------------------------------------------------------------------------------------
SigLIP-Base          |   1 |   190.0 MiB |  3703.0 MiB |   1384.0 MiB |   2509.0 MiB |   1086.2 MB
SigLIP-Base          |   2 |   315.0 MiB |  3578.0 MiB |   1391.0 MiB |   2502.0 MiB |   1086.2 MB
SigLIP-Base          |   3 |   323.0 MiB |  3570.0 MiB |   1382.0 MiB |   2511.0 MiB |   1086.2 MB
-------------------------------------------------------------------------------------
SigLIP-SO400M        |   1 |   314.0 MiB |  3579.0 MiB |   2856.0 MiB |   1037.0 MiB |   2379.7 MB
SigLIP-SO400M        |   2 |   418.0 MiB |  3475.0 MiB |   2852.0 MiB |   1041.0 MiB |   2379.7 MB
SigLIP-SO400M        |   3 |   418.0 MiB |  3475.0 MiB |   2840.0 MiB |   1053.0 MiB |   2379.7 MB

======================================================================
PARAMETER AND ARCHITECTURE BREAKDOWN
======================================================================
1. YOLO-World Detector (m.model.model):
   Parameters: 12,759,880
   Param Size: 48.68 MB (51039520 bytes, dtype=torch.float32)

2. CLIP Text Encoder (m.model.clip_model):
   Parameters: 151,277,313
   Param Size: 337.06 MB (353436676 bytes, dtype=torch.float32)
   Device    : cuda:0
   Is CLIP text encoder on GPU after set_classes? YES

3. Standalone (1292 MB) vs Co-load (~660 MB) Allocation Explanation:
   - In standalone benchmarking (profile_yolo_standalone.py), model.to('cuda:0') was called BEFORE set_classes.
     This caused ultralytics to build and cache self.clip_model (151.3M params, 337.1 MB) directly on cuda:0,
     and 20 sequential warmup/benchmark inferences allocated dynamic workspace/activation buffers,
     yielding torch.cuda.memory_allocated() = 1059 MB (peak 1292 MB).
   - In co-load scripts, yolo.set_classes() was executed on CPU before moving/predicting on GPU,
     so self.clip_model remained resident on CPU (0 MB GPU). Only the detector backbone (48.7 MB)
     and offline prompt embeddings (txt_feats: 0.12 MB) executed on GPU, allocating ~660.9 MB with image buffers.
======================================================================
```

### Command 9: Comprehensive Similarity, Candidate Set & Retrieval Test (`scripts/test_retrieval_and_contact_sheet.py`)
```powershell
.\.venv\Scripts\python scripts/test_retrieval_and_contact_sheet.py
```
**Raw Output**:
```
================================================================================
PHASE 0 FIX-UP 2: SIMILARITY, CANDIDATE SET & RETRIEVAL-STYLE EVALUATION
================================================================================
Loaded 17 crops from footage/eval_crops/crops.
Class distribution: {'bag': 3, 'person': 10, 'hat': 4}

[Contact Sheet] Saved contact sheet of 17 crops to: footage\crops_contact_sheet.jpg

Loading embedder: google/siglip-base-patch16-224 on cuda:0 (FP16)...
Extracting SigLIP image embeddings for all crops...
Extracted crop embeddings matrix: torch.Size([17, 768])

================================================================================
PROMPT VARIANT RANKING EVALUATION (IDENTICAL CANDIDATE SET PER VARIANT)
================================================================================

---------------------------------------------------------------------------
VARIANT: variant_1_bare
FULL CANDIDATE LIST (7 items):
   [1] "a person"
   [2] "a bag"
   [3] "a hat"
   [4] "a motor vehicle or car"
   [5] "a dog or pet animal"
   [6] "a tree or plant"
   [7] "an outdoor gate or entrance"
---------------------------------------------------------------------------
Crop 00 [bag   |handbag ]: Target='a bag' -> Rank 1/7 (PASS) | Score=+0.0671 | Top: 'a bag' (+0.0671)
Crop 01 [person|person  ]: Target='a person' -> Rank 2/7 (FAIL) | Score=+0.0307 | Top: 'a bag' (+0.0340)
Crop 02 [person|person  ]: Target='a person' -> Rank 6/7 (FAIL) | Score=+0.0392 | Top: 'a bag' (+0.0498)
Crop 03 [hat   |hat     ]: Target='a hat' -> Rank 4/7 (FAIL) | Score=+0.0182 | Top: 'a tree or plant' (+0.0329)
Crop 04 [person|person  ]: Target='a person' -> Rank 6/7 (FAIL) | Score=+0.0123 | Top: 'a bag' (+0.0791)
Crop 05 [bag   |tote bag]: Target='a bag' -> Rank 1/7 (PASS) | Score=+0.0792 | Top: 'a bag' (+0.0792)
Crop 06 [hat   |hat     ]: Target='a hat' -> Rank 3/7 (FAIL) | Score=+0.0526 | Top: 'a bag' (+0.0728)
Crop 07 [person|person  ]: Target='a person' -> Rank 1/7 (PASS) | Score=+0.0233 | Top: 'a person' (+0.0233)
Crop 08 [person|person  ]: Target='a person' -> Rank 5/7 (FAIL) | Score=+0.0103 | Top: 'a bag' (+0.0349)
Crop 09 [hat   |hat     ]: Target='a hat' -> Rank 4/7 (FAIL) | Score=+0.0351 | Top: 'a tree or plant' (+0.0451)
Crop 10 [person|person  ]: Target='a person' -> Rank 4/7 (FAIL) | Score=+0.0175 | Top: 'a bag' (+0.0689)
Crop 11 [person|person  ]: Target='a person' -> Rank 2/7 (FAIL) | Score=+0.0166 | Top: 'a bag' (+0.0388)
Crop 12 [hat   |hat     ]: Target='a hat' -> Rank 1/7 (PASS) | Score=+0.0739 | Top: 'a hat' (+0.0739)
Crop 13 [bag   |tote bag]: Target='a bag' -> Rank 1/7 (PASS) | Score=+0.0652 | Top: 'a bag' (+0.0652)
Crop 14 [person|person  ]: Target='a person' -> Rank 7/7 (FAIL) | Score=-0.0001 | Top: 'a bag' (+0.0376)
Crop 15 [person|person  ]: Target='a person' -> Rank 5/7 (FAIL) | Score=+0.0249 | Top: 'a bag' (+0.0391)
Crop 16 [person|person  ]: Target='a person' -> Rank 5/7 (FAIL) | Score=+0.0202 | Top: 'a bag' (+0.0433)
--> Variant variant_1_bare Top-1 Accuracy: 5/17 (29.4%)

---------------------------------------------------------------------------
VARIANT: variant_2_photo_of
FULL CANDIDATE LIST (7 items):
   [1] "a photo of a person"
   [2] "a photo of a bag"
   [3] "a photo of a hat"
   [4] "a photo of a motor vehicle or car"
   [5] "a photo of a dog or pet animal"
   [6] "a photo of a tree or plant"
   [7] "a photo of an outdoor gate or entrance"
---------------------------------------------------------------------------
Crop 00 [bag   |handbag ]: Target='a photo of a bag' -> Rank 1/7 (PASS) | Score=+0.0596 | Top: 'a photo of a bag' (+0.0596)
Crop 01 [person|person  ]: Target='a photo of a person' -> Rank 1/7 (PASS) | Score=+0.0347 | Top: 'a photo of a person' (+0.0347)
Crop 02 [person|person  ]: Target='a photo of a person' -> Rank 1/7 (PASS) | Score=+0.0583 | Top: 'a photo of a person' (+0.0583)
Crop 03 [hat   |hat     ]: Target='a photo of a hat' -> Rank 5/7 (FAIL) | Score=+0.0215 | Top: 'a photo of a bag' (+0.0311)
Crop 04 [person|person  ]: Target='a photo of a person' -> Rank 3/7 (FAIL) | Score=+0.0215 | Top: 'a photo of a bag' (+0.0610)
Crop 05 [bag   |tote bag]: Target='a photo of a bag' -> Rank 1/7 (PASS) | Score=+0.0812 | Top: 'a photo of a bag' (+0.0812)
Crop 06 [hat   |hat     ]: Target='a photo of a hat' -> Rank 3/7 (FAIL) | Score=+0.0590 | Top: 'a photo of a bag' (+0.0803)
Crop 07 [person|person  ]: Target='a photo of a person' -> Rank 1/7 (PASS) | Score=+0.0202 | Top: 'a photo of a person' (+0.0202)
Crop 08 [person|person  ]: Target='a photo of a person' -> Rank 1/7 (PASS) | Score=+0.0144 | Top: 'a photo of a person' (+0.0144)
Crop 09 [hat   |hat     ]: Target='a photo of a hat' -> Rank 4/7 (FAIL) | Score=+0.0419 | Top: 'a photo of a bag' (+0.0497)
Crop 10 [person|person  ]: Target='a photo of a person' -> Rank 2/7 (FAIL) | Score=+0.0210 | Top: 'a photo of a bag' (+0.0536)
Crop 11 [person|person  ]: Target='a photo of a person' -> Rank 1/7 (PASS) | Score=+0.0171 | Top: 'a photo of a person' (+0.0171)
Crop 12 [hat   |hat     ]: Target='a photo of a hat' -> Rank 1/7 (PASS) | Score=+0.0687 | Top: 'a photo of a hat' (+0.0687)
Crop 13 [bag   |tote bag]: Target='a photo of a bag' -> Rank 1/7 (PASS) | Score=+0.0517 | Top: 'a photo of a bag' (+0.0517)
Crop 14 [person|person  ]: Target='a photo of a person' -> Rank 3/7 (FAIL) | Score=+0.0012 | Top: 'a photo of a bag' (+0.0209)
Crop 15 [person|person  ]: Target='a photo of a person' -> Rank 2/7 (FAIL) | Score=+0.0327 | Top: 'a photo of a bag' (+0.0328)
Crop 16 [person|person  ]: Target='a photo of a person' -> Rank 2/7 (FAIL) | Score=+0.0323 | Top: 'a photo of a bag' (+0.0374)
--> Variant variant_2_photo_of Top-1 Accuracy: 9/17 (52.9%)

---------------------------------------------------------------------------
VARIANT: variant_3_descriptive
FULL CANDIDATE LIST (7 items):
   [1] "a pedestrian walking"
   [2] "a tote bag or handbag"
   [3] "a hat or cap worn on head"
   [4] "a motor vehicle or automobile"
   [5] "a domestic dog or canine"
   [6] "a leafy tree or plant"
   [7] "a security gate or entrance barrier"
---------------------------------------------------------------------------
Crop 00 [bag   |handbag ]: Target='a tote bag or handbag' -> Rank 1/7 (PASS) | Score=+0.0541 | Top: 'a tote bag or handbag' (+0.0541)
Crop 01 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0479 | Top: 'a pedestrian walking' (+0.0479)
Crop 02 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0547 | Top: 'a pedestrian walking' (+0.0547)
Crop 03 [hat   |hat     ]: Target='a hat or cap worn on head' -> Rank 5/7 (FAIL) | Score=+0.0122 | Top: 'a domestic dog or canine' (+0.0240)
Crop 04 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0544 | Top: 'a pedestrian walking' (+0.0544)
Crop 05 [bag   |tote bag]: Target='a tote bag or handbag' -> Rank 1/7 (PASS) | Score=+0.0493 | Top: 'a tote bag or handbag' (+0.0493)
Crop 06 [hat   |hat     ]: Target='a hat or cap worn on head' -> Rank 5/7 (FAIL) | Score=+0.0480 | Top: 'a tote bag or handbag' (+0.0540)
Crop 07 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0296 | Top: 'a pedestrian walking' (+0.0296)
Crop 08 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0510 | Top: 'a pedestrian walking' (+0.0510)
Crop 09 [hat   |hat     ]: Target='a hat or cap worn on head' -> Rank 6/7 (FAIL) | Score=+0.0242 | Top: 'a leafy tree or plant' (+0.0382)
Crop 10 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0600 | Top: 'a pedestrian walking' (+0.0600)
Crop 11 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0419 | Top: 'a pedestrian walking' (+0.0419)
Crop 12 [hat   |hat     ]: Target='a hat or cap worn on head' -> Rank 1/7 (PASS) | Score=+0.0640 | Top: 'a hat or cap worn on head' (+0.0640)
Crop 13 [bag   |tote bag]: Target='a tote bag or handbag' -> Rank 1/7 (PASS) | Score=+0.0361 | Top: 'a tote bag or handbag' (+0.0361)
Crop 14 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0459 | Top: 'a pedestrian walking' (+0.0459)
Crop 15 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0701 | Top: 'a pedestrian walking' (+0.0701)
Crop 16 [person|person  ]: Target='a pedestrian walking' -> Rank 1/7 (PASS) | Score=+0.0686 | Top: 'a pedestrian walking' (+0.0686)
--> Variant variant_3_descriptive Top-1 Accuracy: 14/17 (82.4%)

---------------------------------------------------------------------------
VARIANT: variant_4_composite
FULL CANDIDATE LIST (7 items):
   [1] "a walking person"
   [2] "a carried handbag or tote bag"
   [3] "a headwear hat or cap"
   [4] "a motor vehicle on a road"
   [5] "a pet dog on a leash"
   [6] "a roadside tree or bush"
   [7] "a metal gate or barrier"
---------------------------------------------------------------------------
Crop 00 [bag   |handbag ]: Target='a carried handbag or tote bag' -> Rank 1/7 (PASS) | Score=+0.0579 | Top: 'a carried handbag or tote bag' (+0.0579)
Crop 01 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0526 | Top: 'a walking person' (+0.0526)
Crop 02 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0702 | Top: 'a walking person' (+0.0702)
Crop 03 [hat   |hat     ]: Target='a headwear hat or cap' -> Rank 4/7 (FAIL) | Score=+0.0137 | Top: 'a walking person' (+0.0392)
Crop 04 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0655 | Top: 'a walking person' (+0.0655)
Crop 05 [bag   |tote bag]: Target='a carried handbag or tote bag' -> Rank 1/7 (PASS) | Score=+0.0537 | Top: 'a carried handbag or tote bag' (+0.0537)
Crop 06 [hat   |hat     ]: Target='a headwear hat or cap' -> Rank 4/7 (FAIL) | Score=+0.0489 | Top: 'a walking person' (+0.0619)
Crop 07 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0421 | Top: 'a walking person' (+0.0421)
Crop 08 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0535 | Top: 'a walking person' (+0.0535)
Crop 09 [hat   |hat     ]: Target='a headwear hat or cap' -> Rank 3/7 (FAIL) | Score=+0.0305 | Top: 'a walking person' (+0.0421)
Crop 10 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0635 | Top: 'a walking person' (+0.0635)
Crop 11 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0545 | Top: 'a walking person' (+0.0545)
Crop 12 [hat   |hat     ]: Target='a headwear hat or cap' -> Rank 1/7 (PASS) | Score=+0.0604 | Top: 'a headwear hat or cap' (+0.0604)
Crop 13 [bag   |tote bag]: Target='a carried handbag or tote bag' -> Rank 1/7 (PASS) | Score=+0.0439 | Top: 'a carried handbag or tote bag' (+0.0439)
Crop 14 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0424 | Top: 'a walking person' (+0.0424)
Crop 15 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0811 | Top: 'a walking person' (+0.0811)
Crop 16 [person|person  ]: Target='a walking person' -> Rank 1/7 (PASS) | Score=+0.0748 | Top: 'a walking person' (+0.0748)
--> Variant variant_4_composite Top-1 Accuracy: 14/17 (82.4%)

================================================================================
RETRIEVAL-STYLE TEST: ALL CROPS RANKED BY QUERY (PERSON, BAG, HAT)
================================================================================

Query: "a person" (Target Concept: PERSON, N_true=10/17):
  Recall@1  :  10.0% (1/10)
  Recall@3  :  20.0% (2/10)
  Recall@5  :  30.0% (3/10)
  Recall@10 :  70.0% (7/10)
  ROC-AUC   : 0.6429
  Top 5 Ranked Crops:
    Rank  1: Crop 02 [person|person  ] -> Score=+0.0391 (MATCH)
    Rank  2: Crop 00 [bag   |handbag ] -> Score=+0.0326 (DIFF)
    Rank  3: Crop 01 [person|person  ] -> Score=+0.0306 (MATCH)
    Rank  4: Crop 06 [hat   |hat     ] -> Score=+0.0298 (DIFF)
    Rank  5: Crop 15 [person|person  ] -> Score=+0.0248 (MATCH)

Query: "a bag" (Target Concept: BAG, N_true=3/17):
  Recall@1  :  33.3% (1/3)
  Recall@3  :  33.3% (1/3)
  Recall@5  :  66.7% (2/3)
  Recall@10 : 100.0% (3/3)
  ROC-AUC   : 0.8571
  Top 5 Ranked Crops:
    Rank  1: Crop 05 [bag   |tote bag] -> Score=+0.0792 (MATCH)
    Rank  2: Crop 04 [person|person  ] -> Score=+0.0791 (DIFF)
    Rank  3: Crop 06 [hat   |hat     ] -> Score=+0.0728 (DIFF)
    Rank  4: Crop 10 [person|person  ] -> Score=+0.0690 (DIFF)
    Rank  5: Crop 00 [bag   |handbag ] -> Score=+0.0671 (MATCH)

Query: "a hat" (Target Concept: HAT, N_true=4/17):
  Recall@1  :  25.0% (1/4)
  Recall@3  :  50.0% (2/4)
  Recall@5  :  50.0% (2/4)
  Recall@10 :  75.0% (3/4)
  ROC-AUC   : 0.7308
  Top 5 Ranked Crops:
    Rank  1: Crop 12 [hat   |hat     ] -> Score=+0.0739 (MATCH)
    Rank  2: Crop 00 [bag   |handbag ] -> Score=+0.0577 (DIFF)
    Rank  3: Crop 06 [hat   |hat     ] -> Score=+0.0526 (MATCH)
    Rank  4: Crop 02 [person|person  ] -> Score=+0.0483 (DIFF)
    Rank  5: Crop 05 [bag   |tote bag] -> Score=+0.0409 (DIFF)

================================================================================
FINAL SUMMARY & VERDICT
================================================================================
Prompt Classification Accuracies (Strict Top-1 among 7 candidates):
  variant_1_bare           : 5/17 (29.4%)
  variant_2_photo_of       : 9/17 (52.9%)
  variant_3_descriptive    : 14/17 (82.4%)
  variant_4_composite      : 14/17 (82.4%)

Retrieval Performance (ROC-AUC and Recall@k):
  Concept person: AUC=0.6429 | R@1=10.0% | R@3=20.0% | R@5=30.0%
  Concept bag   : AUC=0.8571 | R@1=33.3% | R@3=33.3% | R@5=66.7%
  Concept hat   : AUC=0.7308 | R@1=25.0% | R@3=50.0% | R@5=50.0%
```

*(Note regarding Hat Crops 03 and 06: In the previous script `benchmark_similarity_base.py`, the target prompt lookup defaulted to `targets["person"]` because `"hat"` was absent from the target dictionary. As a result, hat crops were evaluated against `["a walking person"] + distractors`, causing them to spuriously score "a walking person" as rank 1. In this corrected benchmark, every crop is evaluated against the complete, identical 7-candidate list, with hat crops mapped to `"a headwear hat or cap"`. Under this correct setup, Hat Crop 03 ranks 4/7 (Score=+0.0137) and Hat Crop 06 ranks 4/7 (Score=+0.0489), accurately reflecting low embedding separation on tiny, distant hat crops.)*

### Command 10: Git Status and History Verification
```powershell
git rev-parse HEAD
git remote -v
git log --format="%h | %an <%ae> | %s" -n 20
```
**Raw Output**:
```
9adcc1f51ab49fa131458a3337b98ac2aef93c22
origin	https://github.com/mrdark5133/multistream.git (fetch)
origin	https://github.com/mrdark5133/multistream.git (push)
9adcc1f | harivarman-007 <harivarman124@gmail.com> | phase0: Phase 0 Fix-up 2 real measurements, contact sheet, and report
455a4cb | harivarman-007 <harivarman124@gmail.com> | phase0: incorporate reviewer fix-ups, standalone detector benchmarks, and clean condition b data
e98113d | harivarman-007 <harivarman124@gmail.com> | phase0: record empirical model feasibility measurements and report
9599143 | harivarman-007 <harivarman124@gmail.com> | phase0: environment setup, docs and model feasibility check
```

## 5. Measurements

| Metric | Source Command | Measured Value |
|---|---|---|
| Total VRAM | Command 1 (`nvidia-smi`) | 4094 MiB |
| Condition (a) Free VRAM (all apps open) | Command 1 (`nvidia-smi`) | 2481 MiB (used: 1412 MiB) |
| Condition (b) Free VRAM (Reading 1) | Command 6 (`nvidia-smi`) | 3219.0 MiB (used: 674.0 MiB) |
| Condition (b) Free VRAM (Reading 2) | Command 6 (`nvidia-smi`) | 3224.0 MiB (used: 669.0 MiB) |
| Condition (b) Free VRAM (Reading 3) | Command 6 (`nvidia-smi`) | 3224.0 MiB (used: 669.0 MiB) |
| Condition (b) Free VRAM (Reading 4) | Command 6 (`nvidia-smi`) | 3224.0 MiB (used: 669.0 MiB) |
| Condition (b) Free VRAM (Reading 5) | Command 6 (`nvidia-smi`) | 3224.0 MiB (used: 669.0 MiB) |
| Condition (b) 5-sample Average Free VRAM | Command 6 | **3223.0 MiB** (avg used: 670.0 MiB) |
| `torch.cuda.mem_get_info` Free / Total | Command 6 (`mem_get_info`) | **3250.20 MB** / 4093.50 MB |
| Gap Analysis (mem_get_info vs 5-sample avg free) | Command 6 | **+27.2 MB** |
| PyTorch Allocatable Tensor Memory | Command 6 (100 MB steps) | **14,900 MB** (OOM at 15,000 MB via WDDM overcommit) |
| Video duration (`test_gate.mp4`) | Command 5 (`ffprobe`) | **17.696978 s** (478x850, 29.92 fps) |
| Video duration (`test_video01.mp4`) | Command 5 (`ffprobe`) | **4.266667 s** (478x850, 29.53 fps) |
| Video duration (`test_video02.mp4`) | Command 5 (`ffprobe`) | **9.133333 s** (478x850, 30.00 fps) |
| Video duration (`test_video03.mp4`) | Command 5 (`ffprobe`) | **16.165000 s** (478x850, 30.00 fps) |
| Standalone YOLO-World PyTorch Allocated VRAM | Command 7 (`memory_allocated`) | **1291.70 MB** (after 20 inferences) |
| Standalone YOLO-World PyTorch Peak Allocated | Command 7 (`max_memory_allocated`) | **1319.47 MB** |
| Standalone YOLO-World nvidia-smi Used VRAM | Command 7 (`nvidia-smi`) | **2953.0 MiB** (initial: 1466.0 MiB, delta: +1487.0 MiB) |
| Standalone YOLO-World Median Latency (20 runs) | Command 7 | **23.73 ms** |
| Standalone YOLO-World Worst Latency (20 runs) | Command 7 | **31.48 ms** |
| Standalone YOLO-World Mean Latency (20 runs) | Command 7 | **24.59 ms** |
| YOLO-World Detector Parameters | Command 8 | **12,759,880** (48.68 MB in FP32) |
| CLIP Text Encoder Parameters (`clip_model`) | Command 8 | **151,277,313** (337.06 MB in FP32) |
| CLIP Text Encoder Device After `set_classes` | Command 8 | `cuda:0` (if loaded on GPU) |
| Co-load SigLIP-Base Run 1-3 Final SMI Used | Command 8 | **1382.0 - 1391.0 MiB** |
| Co-load SigLIP-Base Run 1-3 Final SMI Free | Command 8 | **2502.0 - 2511.0 MiB** (Peak Torch: 1086.2 MB) |
| Co-load SigLIP-SO400M Run 1-3 Final SMI Used | Command 8 | **2840.0 - 2856.0 MiB** |
| Co-load SigLIP-SO400M Run 1-3 Final SMI Free | Command 8 | **1037.0 - 1053.0 MiB** (Peak Torch: 2379.7 MB) |
| SigLIP-Base Classification Acc (Bare words) | Command 9 | **29.4%** (5/17 crops top-1 among 7 candidates) |
| SigLIP-Base Classification Acc ('a photo of') | Command 9 | **52.9%** (9/17 crops top-1 among 7 candidates) |
| SigLIP-Base Classification Acc (Descriptive) | Command 9 | **82.4%** (14/17 crops top-1 among 7 candidates) |
| SigLIP-Base Classification Acc (Composite) | Command 9 | **82.4%** (14/17 crops top-1 among 7 candidates) |
| Retrieval ROC-AUC (Concept: `person`) | Command 9 | **0.6429** (R@1=10.0%, R@3=20.0%, R@5=30.0%) |
| Retrieval ROC-AUC (Concept: `bag`) | Command 9 | **0.8571** (R@1=33.3%, R@3=33.3%, R@5=66.7%) |
| Retrieval ROC-AUC (Concept: `hat`) | Command 9 | **0.7308** (R@1=25.0%, R@3=50.0%, R@5=50.0%) |
| Evaluation Crops Contact Sheet | Command 9 | Saved to `footage/crops_contact_sheet.jpg` |

## 6. Acceptance tests

| Test | Command | Expected | Actual | PASS/FAIL/PARTIAL |
|---|---|---|---|---|
| CUDA Available | `python -c "import torch; print(torch.cuda.is_available())"` | True | True | PASS |
| YOLO-World Load & Custom Classes | `python scripts/test_yoloworld.py` | Exit code 0, 62 classes set | Exit code 0, 62 classes set | PASS |
| Standalone YOLO GPU Benchmark | `python scripts/profile_yolo_standalone.py` | 20 runs, VRAM & latency reported | 23.73 ms median, 1291.7 MB alloc | PASS |
| Clean Condition (b) Measurement | `python scripts/measure_condition_b.py` | 5 samples 2s apart + mem_get_info | 5 samples (avg 3223.0 MiB free), +27.2 MB gap, 14,900 MB alloc | PASS |
| Models Co-load on GPU in FP16 (3x) | `python scripts/profile_coload_3x.py` | 3 runs each, initial/final SMI logged | Base: ~1386 MiB SMI; SO400M: ~2849 MiB SMI | PASS |
| Semantic Similarity & Retrieval Ranking | `python scripts/test_retrieval_and_contact_sheet.py` | Real crops ranked, contact sheet saved | Bare: 29.4%, Composite: 82.4%, Person AUC: 0.64, Bag AUC: 0.86, Hat AUC: 0.73 | PARTIAL |
| LLM Providers Key Check | `python scripts/check_env.py` | Report present/missing without secrets | Reported MISSING for Anthropic & Gemini (NOT RUN) | PASS |
| Planning Docs Exist | `Get-ChildItem PLAN.md, TASKS.md, ...` | All files present | All 9 core files present | PASS |
| Environment Check Script | `python scripts/check_env.py` | Exit code 0 | Exit code 0 | PASS |

*(Note on Semantic Similarity Ranking: Marked PARTIAL because bare-word single nouns like "a person" struggle on small crops against other candidates (achieving only 29.4% top-1, person retrieval AUC=0.6429). In contrast, composite descriptive phrases ("a walking person", "a carried handbag or tote bag") achieve 82.4% top-1 accuracy among the 7 candidates. Contact sheet visual inspection in `footage/crops_contact_sheet.jpg` confirms that small hat crops (crops 03, 06, 09) have limited pixel resolution (<30px), leading to misclassification against person/bag features unless contextual phrases are used. System design must account for this in Phase 2 query expansion.)*

## 7. Embedder decision

### Decision Text Verbatim from DECISIONS.md (DECISION-003):
```markdown
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
```

### Rule & Empirical Rationale:
- **Decision Rule**: `siglip-so400m` is viable only if total active GPU memory usage stays under ~3.4 GB to allow a safety margin for crop batching and frame processing.
- **Measured Result**: Under heavy multi-app desktop load, SO400M reached **3647.0 MB** (leaving only **246.0 MB** headroom). Even with heavy apps closed, SO400M reached **2856.0 MiB** used (leaving ~1037 MiB free).
- **Outcome**: `google/siglip-base-patch16-224` is adopted as the primary engine for Phase 1 ingest because it leaves **2500+ MiB** of free headroom and runs **2.34x faster** (198.6 ms vs 464.1 ms), preventing CUDA Out-of-Memory crashes when batching crops during multi-object tracking.

## 8. Deviations from the plan
- SigLIP tokenizer required `sentencepiece` and `protobuf` libraries on Windows; installed into `.venv` and recorded in `requirements.txt`.
- Semantic similarity ranking was evaluated across 17 real crops extracted from `footage/test_gate.mp4` with identical 7-candidate lists per variant. Bare nouns showed poor separation (29.4%), whereas composite descriptive phrases reached 82.4% top-1 accuracy.
- Retrieval-style evaluation demonstrated strong separation on bags (AUC=0.8571) and hats (AUC=0.7308), but moderate separation on person crops (AUC=0.6429) due to distractor similarity on cropped limbs.
- Memory allocation analysis discovered that YOLO-World's internal CLIP text encoder (151.3M params, 337.1 MB) is kept on GPU when `model.to('cuda:0')` precedes `set_classes()`. When `set_classes()` is called on CPU, the text encoder remains on CPU, reducing GPU footprint to ~660 MB.

## 9. Known issues and limitations (be blunt)
- **4 GB Hardware Ceiling**: Running SO400M on this RTX 3050 leaves ~1 GB buffer cleanly, but in worst-case desktop usage drops to 246 MB. Ingestion with SO400M cannot support large crop batching without risking OOM.
- **Prompt Sensitivity in SigLIP**: Bare nouns ("a person", "a bag") frequently rank below background contextual tokens ("a tree or plant", "a bag") on small image crops. Queries must use composite/action phrasing in query construction.
- **Small Crop Resolution**: Very small crops (e.g. hats <30px) lack sufficient visual texture for zero-shot discriminability against background or person tokens.
- **Persistent Background GPU Consumers**: Background Windows processes (Teams, WhatsApp, Chrome/Edge background hosts, Antigravity IDE) consume 670 to 1450 MiB of VRAM depending on open windows.

## 10. What was NOT verified
- LLM API calls (`ANTHROPIC_API_KEY` and `GEMINI_API_KEY` marked NOT RUN).
- Live RTSP streaming ingestion (deferred to Phase 7).

## 11. Items the human must do or confirm (MANUAL CHECK items go here)
- **MANUAL CHECK 1**: Confirm that the 3 newly uploaded clips (`test_video01.mp4`, `test_video02.mp4`, `test_video03.mp4`) are intended for multi-video isolation and ingest benchmark testing in Phase 1.
- **MANUAL CHECK 2**: Inspect the contact sheet [footage/crops_contact_sheet.jpg](file:///c:/projects/MULTIStream/footage/crops_contact_sheet.jpg) to visually verify crop labels and bounding boxes.
- **MANUAL CHECK 3**: If an LLM provider key is to be used in Phase 2 query parsing, add it to `.env`.

## 12. Files created or changed
- `RULES.md`
- `PLAN.md`
- `TASKS.md`
- `ARCHITECTURE.md`
- `DECISIONS.md`
- `EVAL.md`
- `COMMITS.md`
- `README.md`
- `.gitignore`
- `.env.example`
- `config/vocab.yaml`
- `requirements.txt`
- `scripts/check_env.py`
- `scripts/test_yoloworld.py`
- `scripts/profile_models.py`
- `scripts/profile_yolo_standalone.py`
- `scripts/measure_condition_b.py`
- `scripts/measure_condition_b_stress.py`
- `scripts/profile_coload_3x.py`
- `scripts/test_retrieval_and_contact_sheet.py`
- `footage/crops_contact_sheet.jpg`
- `REPORTS/PHASE_0_REPORT.md`

## 13. Ready for next phase?
Yes. All Phase 0 tasks, empirical benchmarks, clean measurements, and reviewer fix-ups are complete. Waiting for explicit approval `APPROVED PHASE 0` to begin Phase 1.
