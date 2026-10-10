# MULTIStream Setup & New Footage Guide

This guide ensures you can set up MULTIStream on any machine without dependency conflicts, version mismatches, or manual configuration headaches.

---

## 1. Quick Setup (From a Clean Clone)

If you clone the repository freshly onto any computer:

### Step 1: Create Python 3.11 Virtual Environment
```powershell
# Open PowerShell in the project directory
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Dependencies with CUDA 12.4 Support
To prevent PyTorch CUDA conflicts on Windows, use the pinned index:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```
*(All PyTorch packages automatically link to CUDA 12.4 via `--extra-index-url https://download.pytorch.org/whl/cu124` specified in `requirements.txt`)*

### Step 3: Download Offline Model Weights (One-liner)
Run the automated downloader to fetch required offline TTS & detection models:
```powershell
python scripts/download_weights.py
```
*(Vision models like Ultralytics YOLO and SigLIP feature embedders are automatically downloaded and cached by HuggingFace and PyTorch on first use).*

---

## 2. Re-Identification (Re-ID) Embeddings: How They Work

You **do NOT** need to manually compute Re-ID embeddings. 

* **Automatic Generation**: Whenever you ingest any video (via live phone streaming, web UI upload, or script), the pipeline automatically:
  1. Detects objects (Grounding-DINO / YOLO).
  2. Tracks instances across frames (ByteTrack).
  3. Crops the target objects and passes them through the **SigLIP Vision Embedder** (`google/siglip-base-patch16-224`).
  4. Stores the 768-dimensional normalized embedding vectors directly in `index_mobile/index.db`.
* **Cross-Camera Matching**: When you query the system or check cross-camera movements, cosine similarity is computed directly against the indexed database vectors in milliseconds.

---

## 3. Working with New Footages

When you want to use fresh video footage, follow any of these 3 methods:

### Method A: Single Command Script Ingestion
Place your video in `footage/` (e.g. `footage/gate_entrance.mp4`) and run:
```powershell
python scripts/ingest.py --video footage/gate_entrance.mp4 --camera cam01 --db index_mobile/index.db
```

### Method B: Via the Web Dashboard
1. Start the server:
   ```powershell
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
2. Open `http://localhost:8000` in your browser.
3. Use the **Upload Video** section to select your new video file and specify a camera name. Ingestion and Re-ID indexing will run in the background.

### Method C: Live Phone Camera Streams
1. Start the server:
   ```powershell
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```
2. Open `http://localhost:8000/live`.
3. Enter your phone's IP camera URL (e.g. `http://192.168.1.100:8080/video`) and click **Connect**.
4. When you click **Stop Camera**, the footage is saved to `footage/recorded/` and automatically indexed with Grounding-DINO and Re-ID embeddings!

---

## 4. Resetting Data for a Fresh Session

To clear old videos and start with a blank database:
```powershell
# Remove old recordings and indexes
Remove-Item -Recurse -Force footage\recorded\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force footage\uploads\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force clips\* -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force index_mobile\* -ErrorAction SilentlyContinue

# Ensure clean directories exist
New-Item -ItemType Directory -Force footage\recorded, footage\uploads, clips, index_mobile | Out-Null
```
Your `.venv` environment and model weights stay untouched, so you can immediately begin working with your new footage.
