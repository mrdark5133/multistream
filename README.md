# MULTIStream: Multi-Stream Video Intelligence with Conversational Query

Multi-camera CCTV intelligence system that indexes video feeds using open-vocabulary detection (YOLO-World), tracking (ByteTrack), and vision-language embeddings (SigLIP), enabling natural-language spatial and temporal search with verifiable visual evidence (annotated snapshots and on-demand video clips).

Built for HackNex 2026 (Problem HNX26EPS05).

---

## 1. Prerequisites
- **OS**: Windows / Linux
- **GPU**: NVIDIA GPU (CUDA-compatible, e.g. RTX 3050 4 GB VRAM)
- **Software**: Python 3.10+, `ffmpeg` on PATH, `git` on PATH

---

## 2. Installation & Setup

1. **Clone & Virtual Environment**:
   ```powershell
   git clone <repo_url>
   cd MULTIStream
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables**:
   ```powershell
   cp .env.example .env
   # Edit .env with your API keys if using LLM query parsing / reranking
   ```

4. **Verify Environment**:
   ```powershell
   python scripts/check_env.py
   ```

---

## 3. Usage

### Ingesting Videos
Drop video files into `footage/` or run the ingest CLI directly:
```powershell
python scripts/ingest.py --videos footage/ --out index_siglip/
```

### Running Queries via CLI
```powershell
python scripts/query.py "did a red car pass through the main gate in the last hour?" --index index_siglip/
```

### Starting API & Chat UI
```powershell
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
Open `http://localhost:8000` in your browser to interact with the conversational chat interface and annotate camera polygons.

---

## 4. Running Tests & Benchmarks
```powershell
pytest tests/ -v
python scripts/eval.py --queries eval/queries.json --index index_siglip/
```

---

## 5. Architectural Principles & Rules
See `RULES.md` for non-negotiable verification and honesty standards.
See `PLAN.md` and `ARCHITECTURE.md` for detailed technical specifications.
