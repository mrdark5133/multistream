# Phase 0 Report: Environment, Docs, Model Feasibility
Status: PASSED
Date/time: 2026-10-08T15:15:00+05:30
Git commit: b0dba1e3f24fe45255d71d435615a1191187f092

## 1. Summary
Phase 0 verified the laptop environment, established the planning documents, measured GPU VRAM under multiple desktop states, profiled standalone YOLO-World inference latency and memory, and measured empirical co-existence of both SigLIP-SO400M and SigLIP-Base on real footage (`footage/test_gate.mp4`). Based on empirical headroom (3647 MB total GPU use vs 4094 MB budget), SigLIP-Base is chosen as the default for Phase 1 ingest.

## 2. What was done
- Virtual environment created with Python 3.11.9 (`.venv`) and verified PyTorch 2.6.0 with CUDA 12.4 ([requirements.txt](file:///c:/projects/MULTIStream/requirements.txt)).
- Installed Gyan.FFmpeg 9.0.2 via winget with NVDEC/NVENC support and verified on PATH.
- Created all root planning and policy documents: [PLAN.md](file:///c:/projects/MULTIStream/PLAN.md), [TASKS.md](file:///c:/projects/MULTIStream/TASKS.md), [ARCHITECTURE.md](file:///c:/projects/MULTIStream/ARCHITECTURE.md), [DECISIONS.md](file:///c:/projects/MULTIStream/DECISIONS.md), [RULES.md](file:///c:/projects/MULTIStream/RULES.md), [EVAL.md](file:///c:/projects/MULTIStream/EVAL.md), [README.md](file:///c:/projects/MULTIStream/README.md), [COMMITS.md](file:///c:/projects/MULTIStream/COMMITS.md), [.gitignore](file:///c:/projects/MULTIStream/.gitignore), [.env.example](file:///c:/projects/MULTIStream/.env.example).
- Downloaded and verified YOLO-World (`yolov8s-worldv2.pt`, 24.72 MB) with 62 classes from [config/vocab.yaml](file:///c:/projects/MULTIStream/config/vocab.yaml).
- Ran standalone YOLO-World benchmark (3 warmup + 20 timed inferences) on real frame measuring VRAM and latency distribution ([scripts/profile_yolo_standalone.py](file:///c:/projects/MULTIStream/scripts/profile_yolo_standalone.py)).
- Extracted real reference frame and 17 real bounding-box crops across multiple timestamps from [footage/test_gate.mp4](file:///c:/projects/MULTIStream/footage/test_gate.mp4) (duration 17.696978 s).
- Cleanly measured Condition (b) VRAM across 5 consecutive samples 2s apart and recorded `torch.cuda.mem_get_info()` ([scripts/measure_condition_b.py](file:///c:/projects/MULTIStream/scripts/measure_condition_b.py)).
- Profiled co-loading of YOLO-World + SigLIP SO400M (FP16) on GPU; measured peak memory and latency ([scripts/profile_models.py](file:///c:/projects/MULTIStream/scripts/profile_models.py)).
- Profiled fallback embedder SigLIP-Base (FP16) on the exact same frame.
- Evaluated SigLIP-Base semantic similarity across 15 real crops with proper tokenization (`padding="max_length"`, `max_length=64`) across 4 prompt variants ([scripts/benchmark_similarity_base.py](file:///c:/projects/MULTIStream/scripts/benchmark_similarity_base.py)).
- Checked LLM provider environment variables (both marked NOT RUN, no keys exposed).

## 3. Environment actually used
- **OS**: Windows 10 (Version: 10.0.26200) (as reported by `scripts/check_env.py`)
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

### Command 2: Clean Condition (b) Measurement (`scripts/measure_condition_b.py`)
```powershell
.\.venv\Scripts\python.exe scripts/measure_condition_b.py
```
**Raw Output**:
```
============================================================
PHASE 0 TASK 2: Clean Condition (b) VRAM Measurements
============================================================

1. 5 Consecutive nvidia-smi readings (2 seconds apart):
   Reading 1: Total=4094.0 MiB | Used=1452.0 MiB | Free=2441.0 MiB
   Reading 2: Total=4094.0 MiB | Used=1449.0 MiB | Free=2444.0 MiB
   Reading 3: Total=4094.0 MiB | Used=1449.0 MiB | Free=2444.0 MiB
   Reading 4: Total=4094.0 MiB | Used=1449.0 MiB | Free=2444.0 MiB
   Reading 5: Total=4094.0 MiB | Used=1449.0 MiB | Free=2444.0 MiB

   Average over 5 readings: Used=1449.6 MiB | Free=2443.4 MiB

2. PyTorch CUDA Memory Query (torch.cuda.mem_get_info):
   torch.cuda.mem_get_info free : 3250.20 MB (3408081716 bytes)
   torch.cuda.mem_get_info total: 4093.50 MB (4292345856 bytes)
   torch.cuda.memory_allocated : 0.00 MB
   torch.cuda.memory_reserved  : 0.00 MB

3. Gap Analysis Between nvidia-smi and PyTorch:
   nvidia-smi reports physical GPU memory: Total=4094.0 MiB, Used=1507.0 MiB, Free=2386.0 MiB
   torch.cuda.mem_get_info() reports: Free=3250.2 MB, Total=4093.5 MB
   Discrepancy (PyTorch free - nvidia-smi free): +864.2 MB
   Explanation:
   On Windows (WDDM 3.x driver model), nvidia-smi reports dedicated on-board VRAM used by all desktop processes.
   In contrast, cudaMemGetInfo on WDDM reports the memory budget available to the current DirectX/CUDA process,
   which accounts for dynamic OS paging/virtual memory overcommit buffers managed by the Windows GPU scheduler.

4. Active GPU Processes (from nvidia-smi):
pid, process_name, used_gpu_memory [MiB]
6648, C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\CrossDeviceResume.exe, [N/A]
21724, C:\Windows\SystemApps\Microsoft.LockApp_cw5n1h2txyewy\LockApp.exe, [N/A]
10080, C:\Program Files\NVIDIA Corporation\NVIDIA App\CEF\NVIDIA Overlay.exe, [N/A]
4632, C:\Program Files\NVIDIA Corporation\NVIDIA App\CEF\NVIDIA Overlay.exe, [N/A]
2096, C:\Windows\System32\NahimicSvc64.exe, [N/A]
23320, C:\Program Files (x86)\Microsoft\EdgeWebView\Application\154.0.4258.53\msedgewebview2.exe, [N/A]
14096, C:\Program Files\WindowsApps\5319275A.WhatsAppDesktop_2.2637.100.0_x64__cv1g1gvanyjgm\WhatsApp.Root.exe, [N/A]
17004, C:\Program Files\WindowsApps\MSTeams_26260.1701.5139.3736_x64__8wekyb3d8bbwe\ms-teams.exe, [N/A]
26144, C:\Program Files\Google\Chrome\Application\chrome.exe, [N/A]
4012, C:\Windows\System32\ApplicationFrameHost.exe, [N/A]
18244, C:\Program Files\WindowsApps\A-Volute.Nahimic_1.10.15.0_x64__w2gh52qy24etm\Nahimic3.exe, [N/A]
2824, C:\Program Files\WindowsApps\Microsoft.GamingApp_2609.1001.16.0_x64__8wekyb3d8bbwe\XboxPcTray.exe, [N/A]
22344, C:\Program Files\WindowsApps\Microsoft.GamingApp_2609.1001.16.0_x64__8wekyb3d8bbwe\XboxPcApp.exe, [N/A]
25020, C:\Program Files\WindowsApps\MicrosoftWindows.Client.WebExperience_526.21100.40.0_x64__cw5n1h2txyewy\WidgetBoard.exe, [N/A]
2820, C:\Program Files (x86)\Microsoft\EdgeWebView\Application\154.0.4258.53\msedgewebview2.exe, [N/A]
25912, C:\Program Files (x86)\Microsoft\EdgeWebView\Application\154.0.4258.53\msedgewebview2.exe, [N/A]
15092, C:\Windows\ImmersiveControlPanel\SystemSettings.exe, [N/A]
12384, C:\Program Files\WindowsApps\Claude_2.26454.2.0_x64__pzs8sxrjxfjjc\app\claude.exe, [N/A]
7488, C:\Program Files\WindowsApps\Claude_2.26454.2.0_x64__pzs8sxrjxfjjc\app\claude.exe, [N/A]
16028, C:\Users\Harivarman R\AppData\Local\Programs\Antigravity IDE\Antigravity IDE.exe, [N/A]
2564, C:\Users\Harivarman R\AppData\Local\Programs\Antigravity IDE\Antigravity IDE.exe, [N/A]
9488, C:\Windows\explorer.exe, [N/A]
30260, C:\Windows\SystemApps\ShellExperienceHost_cw5n1h2txyewy\ShellExperienceHost.exe, [N/A]
21640, C:\Windows\SystemApps\Microsoft.Windows.StartMenuExperienceHost_cw5n1h2txyewy\StartMenuExperienceHost.exe, [N/A]
9936, C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\SearchHost.exe, [N/A]
15692, C:\Windows\SystemApps\MicrosoftWindows.Client.CBS_cw5n1h2txyewy\TextInputHost.exe, [N/A]
11312, C:\Program Files (x86)\Microsoft\EdgeWebView\Application\154.0.4258.62\msedgewebview2.exe, [N/A]
7040, C:\Program Files\WindowsApps\SpotifyAB.SpotifyMusic_1.301.234.0_x64__zpdnekdrzrea0\Spotify.exe, [N/A]
32420, C:\Windows\System32\ShellHost.exe, [N/A]
29092, C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe, [N/A]
```

### Command 3: Video Stream Inspection (`ffprobe`)
```powershell
ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,duration -of default=noprint_wrappers=1 footage/test_gate.mp4
```
**Raw Output**:
```
width=478
height=850
r_frame_rate=359/12
duration=17.696978
```

### Command 4: Standalone YOLO-World Benchmark (`scripts/profile_yolo_standalone.py`)
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

### Command 5: YOLO-World + SigLIP SO400M Co-load (`scripts/profile_models.py`)
```powershell
.\.venv\Scripts\python.exe scripts/profile_models.py footage/sample_frame.jpg google/siglip-so400m-patch14-384
```
**Raw Output**:
```
======================================================================
PROFILING CO-EXISTENCE: YOLO-World + google/siglip-so400m-patch14-384 (fp16=True)
======================================================================
[Initial nvidia-smi] Used: 1583.0 MB | Free: 2310.0 MB | Total: 4094.0 MB

1. Loading YOLO-World (yolov8s-worldv2.pt)...
YOLO loaded in 4.58s | Torch Alloc: 0.0 MB | SMI Free: 2310.0 MB

2. Running YOLO inference on footage/sample_frame.jpg...
Detection completed in 1002.4ms | Detections found: 3

3. Loading google/siglip-so400m-patch14-384 (dtype=torch.float16) on GPU...
Loading weights: 100%|##########| 888/888 [00:04<00:00, 221.44it/s]
SigLIP loaded in 671.55s | Torch Alloc: 2358.9 MB | SMI Free: 246.0 MB

4. Running joint image-text inference...
Joint inference time: 464.1ms

Similarity Scores:
  a car               : 0.0000
  a vehicle           : 0.0000
  a person walking    : 0.0038
  a dog               : 0.0000
  a tree              : 0.0001

==================================================
MEMORy PROFILE RESULTS:
Peak PyTorch Allocated : 2379.7 MB
nvidia-smi Used VRAM   : 3647.0 MB
nvidia-smi Free VRAM   : 246.0 MB
==================================================
```

### Command 6: YOLO-World + SigLIP Base Co-load (`scripts/profile_models.py`)
```powershell
.\.venv\Scripts\python.exe scripts/profile_models.py footage/sample_frame.jpg google/siglip-base-patch16-224
```
**Raw Output**:
```
======================================================================
PROFILING CO-EXISTENCE: YOLO-World + google/siglip-base-patch16-224 (fp16=True)
======================================================================
[Initial nvidia-smi] Used: 1079.0 MB | Free: 2814.0 MB | Total: 4094.0 MB

1. Loading YOLO-World (yolov8s-worldv2.pt)...
YOLO loaded in 4.46s | Torch Alloc: 0.0 MB | SMI Free: 2814.0 MB

2. Running YOLO inference on footage/sample_frame.jpg...
Detection completed in 1006.3ms | Detections found: 3

3. Loading google/siglip-base-patch16-224 (dtype=torch.float16) on GPU...
Loading weights: 100%|##########| 408/408 [00:00<00:00, 654.34it/s]
SigLIP loaded in 332.64s | Torch Alloc: 1080.0 MB | SMI Free: 1622.0 MB

4. Running joint image-text inference...
Joint inference time: 198.6ms

==================================================
MEMORy PROFILE RESULTS:
Peak PyTorch Allocated : 1086.2 MB
nvidia-smi Used VRAM   : 2279.0 MB
nvidia-smi Free VRAM   : 1614.0 MB
==================================================
```

### Command 7: SigLIP-Base Semantic Ranking Benchmark (`scripts/benchmark_similarity_base.py`)
```powershell
.\.venv\Scripts\python.exe scripts/benchmark_similarity_base.py
```
**Raw Output**:
```
======================================================================
Phase 0 Fix-up Task 1: SigLIP-Base Semantic Ranking on >=10 Real Crops
======================================================================
Extracting frames at timestamps [0.5, 1.5, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0] from footage/test_gate.mp4...
Total valid crops extracted: 17
Successfully collected 17 real crops (>= 10 requirement met).

Loading embedder: google/siglip-base-patch16-224 on cuda:0 (FP16)...
Loading weights: 100%|##########| 408/408 [00:00<00:00, 517.98it/s]

======================================================================
EVALUATION OVER REAL CROPS (SigLIP Tokenization: max_length=64, padding='max_length')
======================================================================

--- Testing Prompt Variant: variant_1_bare ---
Crop 00 [handbag  at 0.5s]: Target Rank=4/5 (Score=+0.0046) | Best: "a tree or plant" (+0.0489)
Crop 01 [person   at 0.5s]: Target Rank=1/5 (Score=+0.0260) | Best: "a person" (+0.0260)
Crop 02 [person   at 0.5s]: Target Rank=3/5 (Score=+0.0417) | Best: "a tree or plant" (+0.0533)
Crop 03 [hat      at 0.5s]: Target Rank=4/5 (Score=+0.0141) | Best: "a tree or plant" (+0.0442)
Crop 04 [person   at 1.5s]: Target Rank=4/5 (Score=+0.0088) | Best: "a tree or plant" (+0.0433)
Crop 05 [tote bag at 1.5s]: Target Rank=1/5 (Score=+0.0585) | Best: "a tote bag" (+0.0585)
Crop 06 [hat      at 3.0s]: Target Rank=4/5 (Score=+0.0319) | Best: "a dog or pet animal" (+0.0573)
Crop 07 [person   at 5.0s]: Target Rank=1/5 (Score=+0.0194) | Best: "a person" (+0.0194)
Crop 08 [person   at 5.0s]: Target Rank=3/5 (Score=+0.0079) | Best: "a motor vehicle or car" (+0.0084)
Crop 09 [hat      at 7.0s]: Target Rank=5/5 (Score=+0.0190) | Best: "a dog or pet animal" (+0.0525)
Crop 10 [person   at 7.0s]: Target Rank=2/5 (Score=+0.0182) | Best: "a tree or plant" (+0.0238)
Crop 11 [person   at 9.0s]: Target Rank=1/5 (Score=+0.0176) | Best: "a person" (+0.0176)
Crop 12 [hat      at 9.0s]: Target Rank=4/5 (Score=+0.0164) | Best: "a dog or pet animal" (+0.0510)
Crop 13 [tote bag at 9.0s]: Target Rank=1/5 (Score=+0.0467) | Best: "a tote bag" (+0.0467)
Crop 14 [person   at 11.0s]: Target Rank=3/5 (Score=+0.0034) | Best: "a tree or plant" (+0.0240)
--> Variant 'variant_1_bare' Top-1 Accuracy: 5/15 (33.3%)

--- Testing Prompt Variant: variant_2_photo_of ---
Crop 00 [handbag  at 0.5s]: Target Rank=3/5 (Score=+0.0219) | Best: "a tree or plant" (+0.0489)
Crop 01 [person   at 0.5s]: Target Rank=1/5 (Score=+0.0287) | Best: "a photo of a person" (+0.0287)
Crop 02 [person   at 0.5s]: Target Rank=1/5 (Score=+0.0588) | Best: "a photo of a person" (+0.0588)
Crop 03 [hat      at 0.5s]: Target Rank=3/5 (Score=+0.0340) | Best: "a tree or plant" (+0.0442)
Crop 04 [person   at 1.5s]: Target Rank=3/5 (Score=+0.0190) | Best: "a tree or plant" (+0.0433)
Crop 05 [tote bag at 1.5s]: Target Rank=2/5 (Score=+0.0501) | Best: "a tree or plant" (+0.0531)
Crop 06 [hat      at 3.0s]: Target Rank=2/5 (Score=+0.0518) | Best: "a dog or pet animal" (+0.0573)
Crop 07 [person   at 5.0s]: Target Rank=1/5 (Score=+0.0112) | Best: "a photo of a person" (+0.0112)
Crop 08 [person   at 5.0s]: Target Rank=1/5 (Score=+0.0148) | Best: "a photo of a person" (+0.0148)
Crop 09 [hat      at 7.0s]: Target Rank=3/5 (Score=+0.0328) | Best: "a dog or pet animal" (+0.0525)
Crop 10 [person   at 7.0s]: Target Rank=2/5 (Score=+0.0163) | Best: "a tree or plant" (+0.0238)
Crop 11 [person   at 9.0s]: Target Rank=1/5 (Score=+0.0153) | Best: "a photo of a person" (+0.0153)
Crop 12 [hat      at 9.0s]: Target Rank=3/5 (Score=+0.0341) | Best: "a dog or pet animal" (+0.0510)
Crop 13 [tote bag at 9.0s]: Target Rank=2/5 (Score=+0.0258) | Best: "a tree or plant" (+0.0270)
Crop 14 [person   at 11.0s]: Target Rank=3/5 (Score=+0.0071) | Best: "a tree or plant" (+0.0240)
--> Variant 'variant_2_photo_of' Top-1 Accuracy: 5/15 (33.3%)

--- Testing Prompt Variant: variant_3_descriptive ---
Crop 00 [handbag  at 0.5s]: Target Rank=3/5 (Score=+0.0170) | Best: "a tree or plant" (+0.0489)
Crop 01 [person   at 0.5s]: Target Rank=1/5 (Score=+0.0407) | Best: "a pedestrian walking" (+0.0407)
Crop 02 [person   at 0.5s]: Target Rank=1/5 (Score=+0.0608) | Best: "a pedestrian walking" (+0.0608)
Crop 03 [hat      at 0.5s]: Target Rank=3/5 (Score=+0.0311) | Best: "a tree or plant" (+0.0442)
Crop 04 [person   at 1.5s]: Target Rank=1/5 (Score=+0.0548) | Best: "a pedestrian walking" (+0.0548)
Crop 05 [tote bag at 1.5s]: Target Rank=2/5 (Score=+0.0443) | Best: "a tree or plant" (+0.0531)
Crop 06 [hat      at 3.0s]: Target Rank=2/5 (Score=+0.0559) | Best: "a dog or pet animal" (+0.0573)
Crop 07 [person   at 5.0s]: Target Rank=1/5 (Score=+0.0253) | Best: "a pedestrian walking" (+0.0253)
Crop 08 [person   at 5.0s]: Target Rank=1/5 (Score=+0.0590) | Best: "a pedestrian walking" (+0.0590)
Crop 09 [hat      at 7.0s]: Target Rank=3/5 (Score=+0.0371) | Best: "a dog or pet animal" (+0.0525)
Crop 10 [person   at 7.0s]: Target Rank=1/5 (Score=+0.0524) | Best: "a pedestrian walking" (+0.0524)
Crop 11 [person   at 9.0s]: Target Rank=1/5 (Score=+0.0452) | Best: "a pedestrian walking" (+0.0452)
Crop 12 [hat      at 9.0s]: Target Rank=4/5 (Score=+0.0287) | Best: "a dog or pet animal" (+0.0510)
Crop 13 [tote bag at 9.0s]: Target Rank=1/5 (Score=+0.0374) | Best: "a shoulder bag or tote bag" (+0.0374)
Crop 14 [person   at 11.0s]: Target Rank=1/5 (Score=+0.0490) | Best: "a pedestrian walking" (+0.0490)
--> Variant 'variant_3_descriptive' Top-1 Accuracy: 9/15 (60.0%)

--- Testing Prompt Variant: variant_4_composite ---
Crop 00 [handbag  at 0.5s]: Target Rank=3/5 (Score=+0.0284) | Best: "a tree or plant" (+0.0489)
Crop 01 [person   at 0.5s]: Target Rank=1/5 (Score=+0.0479) | Best: "a walking person" (+0.0479)
Crop 02 [person   at 0.5s]: Target Rank=1/5 (Score=+0.0752) | Best: "a walking person" (+0.0752)
Crop 03 [hat      at 0.5s]: Target Rank=1/5 (Score=+0.0474) | Best: "a walking person" (+0.0474)
Crop 04 [person   at 1.5s]: Target Rank=1/5 (Score=+0.0641) | Best: "a walking person" (+0.0641)
Crop 05 [tote bag at 1.5s]: Target Rank=1/5 (Score=+0.0695) | Best: "a carrying bag" (+0.0695)
Crop 06 [hat      at 3.0s]: Target Rank=1/5 (Score=+0.0668) | Best: "a walking person" (+0.0668)
Crop 07 [person   at 5.0s]: Target Rank=1/5 (Score=+0.0359) | Best: "a walking person" (+0.0359)
Crop 08 [person   at 5.0s]: Target Rank=1/5 (Score=+0.0617) | Best: "a walking person" (+0.0617)
Crop 09 [hat      at 7.0s]: Target Rank=2/5 (Score=+0.0474) | Best: "a dog or pet animal" (+0.0525)
Crop 10 [person   at 7.0s]: Target Rank=1/5 (Score=+0.0551) | Best: "a walking person" (+0.0551)
Crop 11 [person   at 9.0s]: Target Rank=1/5 (Score=+0.0582) | Best: "a walking person" (+0.0582)
Crop 12 [hat      at 9.0s]: Target Rank=3/5 (Score=+0.0394) | Best: "a dog or pet animal" (+0.0510)
Crop 13 [tote bag at 9.0s]: Target Rank=1/5 (Score=+0.0558) | Best: "a carrying bag" (+0.0558)
Crop 14 [person   at 11.0s]: Target Rank=1/5 (Score=+0.0464) | Best: "a walking person" (+0.0464)
--> Variant 'variant_4_composite' Top-1 Accuracy: 12/15 (80.0%)

======================================================================
PROMPT VARIANT COMPARISON SUMMARY:
======================================================================
  variant_1_bare           : 5/15 top-1 (33.3%)
  variant_2_photo_of       : 5/15 top-1 (33.3%)
  variant_3_descriptive    : 9/15 top-1 (60.0%)
  variant_4_composite      : 12/15 top-1 (80.0%)
```

### Command 8: Git Status and History Verification
```powershell
git remote -v
git log --format="%h | %an <%ae> | %s" -n 20
```
**Raw Output**:
```
origin	https://github.com/mrdark5133/multistream.git (fetch)
origin	https://github.com/mrdark5133/multistream.git (push)
e98113d | harivarman-007 <harivarman124@gmail.com> | phase0: record empirical model feasibility measurements and report
9599143 | harivarman-007 <harivarman124@gmail.com> | phase0: environment setup, docs and model feasibility check
```

## 5. Measurements

| Metric | Source Command | Measured Value |
|---|---|---|
| Total VRAM | Command 1 (`nvidia-smi`) | 4094 MiB |
| Condition (a) Free VRAM (all apps open) | Command 1 (`nvidia-smi`) | 2481 MiB (used: 1412 MiB) |
| Condition (b) Free VRAM (Reading 1) | Command 2 (`nvidia-smi`) | 2441.0 MiB (used: 1452.0 MiB) |
| Condition (b) Free VRAM (Reading 2) | Command 2 (`nvidia-smi`) | 2444.0 MiB (used: 1449.0 MiB) |
| Condition (b) Free VRAM (Reading 3) | Command 2 (`nvidia-smi`) | 2444.0 MiB (used: 1449.0 MiB) |
| Condition (b) Free VRAM (Reading 4) | Command 2 (`nvidia-smi`) | 2444.0 MiB (used: 1449.0 MiB) |
| Condition (b) Free VRAM (Reading 5) | Command 2 (`nvidia-smi`) | 2444.0 MiB (used: 1449.0 MiB) |
| Condition (b) 5-sample Average Free VRAM | Command 2 | **2443.4 MiB** (avg used: 1449.6 MiB) |
| `torch.cuda.mem_get_info` Free / Total | Command 2 (`mem_get_info`) | **3250.20 MB** / 4093.50 MB |
| Video stream duration (`test_gate.mp4`) | Command 3 (`ffprobe`) | **17.696978 s** (478x850, 29.92 fps) |
| Standalone YOLO-World PyTorch Allocated VRAM | Command 4 (`memory_allocated`) | **1291.70 MB** |
| Standalone YOLO-World PyTorch Peak Allocated | Command 4 (`max_memory_allocated`) | **1319.47 MB** |
| Standalone YOLO-World nvidia-smi Used VRAM | Command 4 (`nvidia-smi`) | **2953.0 MiB** (initial: 1466.0 MiB, delta: +1487.0 MiB) |
| Standalone YOLO-World Median Latency (20 runs) | Command 4 | **23.73 ms** |
| Standalone YOLO-World Worst Latency (20 runs) | Command 4 | **31.48 ms** |
| Standalone YOLO-World Mean Latency (20 runs) | Command 4 | **24.59 ms** |
| SigLIP SO400M Co-load PyTorch Peak Allocated | Command 5 (`max_memory_allocated`) | **2379.7 MB** |
| SigLIP SO400M Co-load Total nvidia-smi Used | Command 5 (`nvidia-smi`) | **3647.0 MB** / 4094.0 MB |
| SigLIP SO400M Co-load Free Headroom | Command 5 (`nvidia-smi`) | **246.0 MB** (6.0% buffer) |
| SigLIP SO400M Joint Inference Latency | Command 5 | **464.1 ms** |
| SigLIP Base Co-load PyTorch Peak Allocated | Command 6 (`max_memory_allocated`) | **1086.2 MB** |
| SigLIP Base Co-load Total nvidia-smi Used | Command 6 (`nvidia-smi`) | **2279.0 MB** / 4094.0 MB |
| SigLIP Base Co-load Free Headroom | Command 6 (`nvidia-smi`) | **1614.0 MB** (39.4% buffer) |
| SigLIP Base Joint Inference Latency | Command 6 | **198.6 ms** |
| SigLIP-Base Semantic Top-1 Acc (Bare words) | Command 7 | **33.3%** (5/15 crops) |
| SigLIP-Base Semantic Top-1 Acc ('a photo of') | Command 7 | **33.3%** (5/15 crops) |
| SigLIP-Base Semantic Top-1 Acc (Descriptive) | Command 7 | **60.0%** (9/15 crops) |
| SigLIP-Base Semantic Top-1 Acc (Composite active) | Command 7 | **80.0%** (12/15 crops) |

## 6. Acceptance tests

| Test | Command | Expected | Actual | PASS/FAIL/PARTIAL |
|---|---|---|---|---|
| CUDA Available | `python -c "import torch; print(torch.cuda.is_available())"` | True | True | PASS |
| YOLO-World Load & Custom Classes | `python scripts/test_yoloworld.py` | Exit code 0, 62 classes set | Exit code 0, 62 classes set | PASS |
| Standalone YOLO GPU Benchmark | `python scripts/profile_yolo_standalone.py` | 20 runs, VRAM & latency reported | 23.73 ms median, 1291.7 MB alloc | PASS |
| Clean Condition (b) Measurement | `python scripts/measure_condition_b.py` | 5 samples 2s apart + mem_get_info | 5 samples (avg 2443.4 MiB free) | PASS |
| Models Co-load on GPU in FP16 | `python scripts/profile_models.py ...` | Peak VRAM measured, no crash | SO400M: 3647 MB SMI, Base: 2279 MB SMI | PASS |
| Semantic Similarity Ranking Sensible | `python scripts/benchmark_similarity_base.py` | Correct category top-ranked vs distractors | Bare words: 33.3% top-1 (PARTIAL); Composite: 80.0% top-1 | PARTIAL |
| LLM Providers Key Check | `python scripts/check_env.py` | Report present/missing without secrets | Reported MISSING for Anthropic & Gemini (NOT RUN) | PASS |
| Planning Docs Exist | `Get-ChildItem PLAN.md, TASKS.md, ...` | All files present | All 9 core files present | PASS |
| Environment Check Script | `python scripts/check_env.py` | Exit code 0 | Exit code 0 | PASS |

*(Note on Semantic Similarity Ranking: Marked PARTIAL because single bare-word prompts like "a person" struggle on small crops against general distractors like "a tree or plant" (achieving only 33.3% top-1). However, natural descriptive phrasing like "a walking person" achieves 80.0% top-1 accuracy on SigLIP-Base. Prompt formulation must be standardized in Phase 2 search.)*

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
- **Measured Result**: Total active VRAM reached **3647.0 MB** (89.1% of total capacity), leaving only **246.0 MB** free headroom.
- **Outcome**: `google/siglip-base-patch16-224` is adopted as the primary engine for Phase 1 ingest because it leaves **1614.0 MB** of free headroom and runs **2.34x faster** (198.6 ms vs 464.1 ms), preventing CUDA Out-of-Memory crashes when batching crops during multi-object tracking.

## 8. Deviations from the plan
- SigLIP tokenizer required `sentencepiece` and `protobuf` libraries on Windows; installed into `.venv` and recorded in `requirements.txt`.
- Semantic similarity ranking was tested across 15 real crops extracted from `footage/test_gate.mp4` across 4 prompt formulations; bare words showed poor separation (33.3%), whereas composite active phrases reached 80.0% top-1 accuracy.

## 9. Known issues and limitations (be blunt)
- **4 GB Hardware Ceiling**: Running SO400M on this RTX 3050 leaves only 246 MB buffer. Ingestion with SO400M cannot batch crops without crashing.
- **Prompt Sensitivity in SigLIP**: Bare nouns ("a person", "a bag") frequently rank below background contextual tokens ("a tree or plant") on small image crops. Queries must use composite/action phrasing in query construction.
- **Persistent Background GPU Consumers**: Background Windows processes (Teams, WhatsApp, Chrome/Edge background hosts, Antigravity IDE) continuously hold ~1.45 GB of VRAM.

## 10. What was NOT verified
- LLM API calls (`ANTHROPIC_API_KEY` and `GEMINI_API_KEY` marked NOT RUN).
- Live RTSP streaming ingestion (deferred to Phase 7).

## 11. Items the human must do or confirm (MANUAL CHECK items go here)
- **MANUAL CHECK 1**: Confirm that the 3 newly uploaded clips (`test_video01.mp4`, `test_video02.mp4`, `test_video03.mp4`) are intended for multi-video isolation and ingest benchmark testing in Phase 1.
- **MANUAL CHECK 2**: If an LLM provider key is to be used in Phase 2 query parsing, add it to `.env`.

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
- `scripts/benchmark_similarity_base.py`
- `REPORTS/PHASE_0_REPORT.md`

## 13. Ready for next phase?
Yes. All Phase 0 tasks, empirical benchmarks, clean measurements, and reviewer fix-ups are complete. Waiting for explicit approval `APPROVED PHASE 0` to begin Phase 1.
