# ANTIGRAVITY MASTER PROMPT: HNX26EPS05 Multi-Stream Video Intelligence

Paste everything below the line into the Antigravity agent as the first message.

---

## 0. ROLE AND MISSION

You are the build agent for a 24-hour hackathon project (HackNex 2026, problem HNX26EPS05: "Multi-Stream Video Intelligence with Conversational Query").

The system ingests recorded (and optionally live) video from several cameras, indexes the objects it sees, and lets a user ask plain-English questions such as "did a red car pass through the main gate in the last hour?". Every answer must resolve to a **specific camera + timestamp + visual evidence** (snapshot and clip). An answer with no traceable source counts as a failure.

You will work **phase by phase**. You will **stop at the end of every phase**, write a report, and **wait for the human to reply "APPROVED PHASE N"** before starting the next phase. The human will give your reports to another reviewer (Claude) who will check them, so every report must be truthful and verifiable.

---

## 1. NON-NEGOTIABLE RULES

Create `RULES.md` in the repo root with this section copied verbatim, and obey it for the whole project.

### 1.1 Honesty rules (most important)
1. **No fake results.** Never invent, estimate, round up, or "expect" a number, test result, log line, accuracy, latency, VRAM figure, or file size. Every number in a report must come from a command you actually ran, and the raw output must be pasted into the report.
2. **No silent mocking.** If you use synthetic, random, or placeholder data for any test, label it **SYNTHETIC** in the report, and never count it toward accuracy or performance claims.
3. **Say "NOT RUN" or "NOT VERIFIED"** for anything you did not execute. Never write "should work", "works", or "passes" about something you did not run.
4. **Do not create ground truth.** Evaluation labels (`eval/queries.json`) are written by the human team. You may build the harness but must not write labels, and must not tune on the held-out split.
5. **If a test fails, report the failure** with the full error text. Do not delete the test, weaken it, or hide it. A failed phase is reported as FAILED or PARTIAL.
6. **Never claim a hardware or model capability from memory.** Check it (for example by running `nvidia-smi`, loading the model, reading file sizes).
7. If something in this prompt is wrong or impossible (for example a library API differs), say so in the report under "Deviations" instead of working around it silently.

### 1.2 Browser and screenshot rules
1. **Do NOT use the browser tool or browser agent.** Do not auto-open a browser, do not take screenshots, do not record screen captures, and do not run browser-based verification.
2. Verify everything through the terminal: unit tests, `pytest`, `curl`/`httpx` calls to the API, SQLite queries, file listings, logs.
3. If a UI behavior truly needs a human to look at it, write a **MANUAL CHECK** item in the report (exact URL, exact steps, expected result) and leave it for the human. Mark it NOT VERIFIED until the human confirms.

### 1.3 Working rules
1. Follow the phases in order. Do not start a phase early. Do not build features from later phases.
2. Keep changes small and reviewable. Commit at the end of every phase with a clear message, and put the commit hash in the report.
3. Never commit secrets. API keys come from environment variables only. Provide `.env.example`.
4. Never commit large files: models, videos, snapshots, the index. Add them to `.gitignore`.
5. Ask the human a question (and stop) if you need something only they can provide: footage, API key, labeled queries, a decision.
6. Prefer simple code that works over clever code. Add logging for every pipeline stage.
7. Do not delete or overwrite user footage or an existing index. Use new output folders.

---

## 2. HARDWARE AND ENVIRONMENT FACTS

These come from the human. Verify them yourself in Phase 0 and report the real values.

- Laptop GPU: NVIDIA RTX 3050, **4 GB VRAM**. System RAM: **24 GB**.
- OS: detect it and report it (do not assume).
- Python 3.10 or 3.11 preferred. `git` and `ffmpeg` must be on PATH.
- Models are downloaded on first run, and the human's internet may be slow. Report download sizes you actually observe.

---

## 3. PROJECT SUMMARY AND WHAT IS BEING JUDGED

Judging rubric (the build is optimized for this):

| Criterion | Weight |
|---|---|
| Retrieval accuracy on held-out natural-language queries vs. baseline | 30% |
| Correctness of camera + timestamp localization | 20% |
| Clarify-once memory (asks when it does not know a referent, never re-asks, persists across restart) | 20% |
| Query latency vs. baseline | 10% |
| Research contribution (what is new vs. baseline, shown by an ablation or comparison) | 20% |
| Bonus: live ingestion, cross-camera tracking, alerts, privacy | extra |

Minimum bar: recorded multi-camera footage (not live); natural-language queries that return the correct camera + timestamp + clip; beat the baseline on retrieval accuracy for held-out queries; clarify-once memory; short write-up.

At judging, judges bring **unseen recorded footage** and **unseen queries**. So ingestion must work on arbitrary video files quickly, and the vocabulary must not be limited to a few fixed classes.

---

## 4. ARCHITECTURE (DECIDED, DO NOT CHANGE WITHOUT REPORTING)

### 4.1 Two phases of work
**Ingest (heavy, once per video):**
video file -> sample frames -> open-vocabulary detection -> tracking -> pick best frame per track -> save annotated snapshot -> embed best crop -> write row to index. Also embed whole frames at a low rate as a fallback.

**Query (fast, every question):**
question -> parse (object description, location referent, time range) -> resolve referent through the alias memory (or ask once) -> filter by camera/time/region -> vector search over track embeddings and frame embeddings -> merge duplicates -> return camera + timestamp + snapshot + clip.

Nothing is detected at query time.

### 4.2 Models
| Role | Model | Notes |
|---|---|---|
| Open-vocabulary detector | YOLO-World via Ultralytics, `yolov8s-worldv2.pt` | Upgrade to `yolov8m-worldv2.pt` only if measured recall is poor |
| Tracker | ByteTrack (built into Ultralytics) | Track IDs must not leak between videos (see Phase 1 tests) |
| Embedder | `google/siglip-so400m-patch14-384` (about 3.5 GB download, 1152-dim) | The human chose this. **4 GB VRAM may not fit it.** Phase 0 measures this. Fallback: `google/siglip-base-patch16-224` (768-dim) |
| Query parser | Pluggable provider: Claude (Anthropic API), Gemini (Google API), or rule-based | Selected by env `PARSER_PROVIDER` = `claude` or `gemini` or `rules`. Keys from `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`; model names from `CLAUDE_MODEL` and `GEMINI_MODEL`. The rule-based parser must always work with no key and is the final fallback |
| Optional rerank (Phase 6) | Pluggable vision-language provider (Gemini or Claude) on the top 10 snapshots | Selected by env `RERANK_PROVIDER` = `gemini` or `claude` or `off`. Query-time only |

Embedder rules:
- Image embedding runs on the GPU in fp16 if it fits, otherwise decide with measured numbers.
- Text embedding for queries may run on CPU to save VRAM. Measure its latency.
- The embedding dimension, model name, and model dtype are written to a `config` table in the index. **Never mix embeddings from two models in one index.** Use one index folder per embedder.
- Do not assume how the Hugging Face SigLIP class is laid out (for example whether you can delete the text tower). Inspect the loaded model and verify any trick you use by comparing embeddings before and after (cosine similarity close to 1.0), and report the real result.

LLM provider rules (Claude and Gemini):
- Put both behind one small interface in `src/llm/providers.py`: `parse(text) -> dict` and `judge_image(image_path, question) -> {answer, confidence}`. The rest of the code never imports a provider SDK directly.
- **Never invent model names, SDK method names, or limits from memory.** Look up the current model name and SDK usage in the provider's official documentation or by listing available models through the API, then record the exact model string you used in `DECISIONS.md` and in the report. If a call fails, report the real error text.
- Provider choice is settled by **measured results**, not preference: on the same set of test queries, report parse accuracy (valid JSON, correct fields) and latency for each provider that has a key. Do not claim one is better without those numbers.
- Free-tier or rate limits may apply. If you hit a rate limit or quota error, report it verbatim, add retry with backoff, and report how it affected latency. Do not hide retries inside reported latency numbers: report them separately.
- Provider failure must degrade gracefully: if the chosen provider errors or times out, fall back to the next one (for example claude -> gemini -> rules) and **log which provider actually answered** in every result and in every report table.
- API keys come only from environment variables. Never print them, log them, or commit them. When checking that a key exists, report only "present" or "missing".
- **Privacy note to track:** a Claude or Gemini call sends text to a cloud service. A rerank call also sends **snapshot images** to the cloud. Keep a running list of exactly what data leaves the machine for each provider, and put it in `ARCHITECTURE.md` and in the Phase 7 privacy notes.

### 4.3 Open vocabulary (important)
The earlier draft used a 10-class list. That is **not acceptable**. Required:
1. A detector vocabulary of about **60 common CCTV-relevant classes** (people, vehicles of all kinds, bags, helmets, umbrellas, animals, common carried objects, and similar). Put it in `config/vocab.yaml`.
2. **Whole-frame embeddings** at about 1 frame every 2 seconds, stored in a `frames` table, as a fallback for scene-level queries and objects the detector missed.
3. Search runs over both tracks and frames. Results are merged and de-duplicated.

### 4.4 Storage rules
- One snapshot and one embedding **per tracked object**, not per detection. Keep the best frame (largest area x confidence, replaced only when clearly better).
- Snapshot: annotated with bounding box, camera name, and absolute timestamp, resized to about 640 px wide, JPEG.
- Store **pointers, not clips**: `video_path` plus `offset_start` and `offset_end` in seconds. Cut clips on demand with ffmpeg.
- Store absolute wall-clock time: `absolute = video_start + offset`.
- Evidence (the snapshot) is created at ingest, so it still exists if raw video is deleted later.

### 4.5 Index schema (SQLite, WAL mode)
Create at least these tables. You may add columns, but report every change.
- `config(key, value)`: embedder name, dimension, dtype, detector model, vocab hash, stride, created_at.
- `videos(path PRIMARY KEY, camera, start_time, fps, width, height, duration_s, start_source, indexed_at)`. `start_source` records how the start time was found.
- `tracks(id, video, camera, track_id, label, t_start, t_end, t_best, offset_start, offset_end, bbox_px, bbox_norm, snapshot, emb)`.
- `frames(id, video, camera, t_abs, offset_s, snapshot, emb)`.
- `aliases(name, camera, polygon_norm NULL, created_at)`: the clarify-once memory.
- Index on `(camera, t_start, t_end)`.

### 4.6 Video start time resolution (fallback chain)
1. Entry in `manifest.json`.
2. Filename pattern `<camera>_YYYYmmddTHHMMSS.mp4`.
3. `ffprobe` creation_time metadata.
4. If still unknown: report it as `UNKNOWN_START` and require the human to supply one. Never guess silently. Store `start_source` for every video.

### 4.7 "Now" for relative time
Judges use recorded footage, so "last hour" must be relative to the **latest timestamp in the index**, not the wall clock. Provide an override parameter.

### 4.8 Clarify-once memory
- If the query mentions a location referent (for example "main gate") that is neither a known camera name nor in `aliases`, the system must **not guess**. It returns a clarification request listing the cameras.
- The user answers by choosing a camera and **optionally drawing a region (polygon) on that camera's snapshot**. The answer is saved to `aliases` immediately and permanently.
- It must never ask again for the same referent, including after a server restart.
- When a polygon is stored, filter tracks by whether the bbox center falls inside the polygon (normalized coordinates).
- Matching of alias names is normalized (case, whitespace, simple plural). Report what normalization you implemented.

### 4.9 Intake modes (both must work, same index)
- **Manual:** video files dropped into `footage/`.
- **Live (stretch, Phase 7):** phones stream over Wi-Fi (RTSP), a recorder script saves 60-second files into `live_footage/` with the start time in the filename.
- One watcher script indexes new files from both folders exactly once, skips files still being written, and survives restarts.

---

## 5. FILES YOU MUST CREATE AT THE START

Create these in the repo root in Phase 0 and keep them updated. They are the single source of truth.

1. **`PLAN.md`**: the full plan (goal, architecture, models, schema, phases, risks). Derive it from this prompt.
2. **`TASKS.md`**: a checklist per phase with checkboxes. Tick a box only when the acceptance test for that task passed, and link to the report section as proof.
3. **`ARCHITECTURE.md`**: module layout, data flow, schema, API contract. Update as the code changes.
4. **`DECISIONS.md`**: every decision with date, reason, and alternatives (for example the embedder choice and why).
5. **`RULES.md`**: Section 1 of this prompt, verbatim.
6. **`EVAL.md`**: evaluation protocol: query format, metrics, baseline definition, ablation rows, how latency is measured.
7. **`REPORTS/`** folder with one file per phase: `PHASE_0_REPORT.md`, `PHASE_1_REPORT.md`, and so on.
8. **`README.md`**: how to install, run ingest, run the server, run tests.
9. **`.env.example`, `.gitignore`, `requirements.txt`**. `.env.example` must list (with empty values) `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `CLAUDE_MODEL`, `GEMINI_MODEL`, `PARSER_PROVIDER`, `RERANK_PROVIDER`.

Suggested layout (adjust if needed, and report changes):

```
config/vocab.yaml
src/ingest/ (detect_track.py, embed.py, start_time.py, pipeline.py)
src/index/ (db.py, schema.sql)
src/query/ (parser.py, timeparse.py, search.py, clips.py)
src/memory/ (aliases.py)
src/api/ (main.py)
src/ui/ (static chat UI)
scripts/ (ingest.py, watch.py, record.py, eval.py)
tests/
eval/ (queries.json provided by human, results/)
footage/  live_footage/  index_<embedder>/   (git-ignored)
REPORTS/
```

---

## 6. REPORT FORMAT (REQUIRED AT THE END OF EVERY PHASE)

Write `REPORTS/PHASE_N_REPORT.md` using exactly this structure, then print its path and stop.

```
# Phase N Report: <title>
Status: PASSED | PARTIAL | FAILED
Date/time: <from the system clock>
Git commit: <hash>

## 1. Summary (5 lines max)
## 2. What was done (bullet list, each item references a file path)
## 3. Environment actually used
   OS, Python version, torch version, CUDA available (real output), GPU name, ultralytics/transformers versions
## 4. Commands run and RAW output
   Paste the real commands and real terminal output (trim only obvious repetition and say that you trimmed).
## 5. Measurements
   A table. Every number must point to the command in section 4 that produced it.
## 6. Acceptance tests
   Table: test | command | expected | actual | PASS/FAIL/NOT RUN
## 7. Deviations from the plan
## 8. Known issues and limitations (be blunt)
## 9. What was NOT verified
## 10. Items the human must do or confirm (MANUAL CHECK items go here)
## 11. Files created or changed (list)
## 12. Ready for next phase? (yes/no, and what blocks it)
```

After writing the report: update `TASKS.md`, commit, then write in chat: "PHASE N COMPLETE (status). Report: REPORTS/PHASE_N_REPORT.md. Waiting for APPROVED PHASE N." and **stop**.

---

## 7. PHASES

### PHASE 0: Environment, docs, model feasibility

Goal: working environment, all planning docs, and a measured answer to "does so400m fit on a 4 GB GPU?"

Tasks:
1. Inspect the machine: OS, Python, `nvidia-smi`, free disk space, `ffmpeg -version`, `git --version`. Create a virtual environment.
2. Create `requirements.txt` and install. Make sure torch is the **CUDA build**. Report `torch.cuda.is_available()` and the GPU name from real output. If CUDA is false, stop and tell the human how to fix it.
3. Create all files from Section 5 (PLAN, TASKS, ARCHITECTURE, DECISIONS, RULES, EVAL, README, .gitignore, .env.example, REPORTS/).
4. Download and load YOLO-World (`yolov8s-worldv2.pt`) and run `set_classes` with the vocabulary in `config/vocab.yaml` (about 60 classes). Report the real download sizes and any errors (note: Ultralytics may need `git` and internet for the CLIP text model).
5. Load `google/siglip-so400m-patch14-384` in fp16 on the GPU **together with** the detector. Measure real peak memory with `torch.cuda.max_memory_allocated()` and, in a second terminal or via subprocess, `nvidia-smi` memory readings. Use a **real image** from the human's footage or any real photo; do not use random noise for the measurement. If no footage exists yet, ask the human for one image and stop.
6. If so400m does not fit or crashes: try the options in this order, measuring each: image-tower only on GPU, smaller batch, then fall back to `siglip-base-patch16-224`. Report each attempt with raw output. Record the final choice and reason in `DECISIONS.md`. Do not choose by guess.
7. Embed a few real images and show that image-text similarity ranks sensibly (for example an image of a car should score higher for "a car" than for "a dog"). Show the raw scores.
8. Check the LLM providers. For each of `ANTHROPIC_API_KEY` and `GEMINI_API_KEY`, report only whether it is present or missing (never print the key). For each key that is present, make one real minimal text call, record the exact model string that worked, the round-trip time, and any error text. If a key is missing, mark that provider NOT RUN and ask the human. Do not guess model names: find the current ones from the provider documentation or a model-listing call.
9. Write a minimal `scripts/check_env.py` that prints all of this in one run (the provider check must print only present/missing and results, never keys).

Acceptance tests: CUDA true; both models load together; measured VRAM recorded; embedder choice recorded with evidence; all docs exist; `scripts/check_env.py` runs.

### PHASE 1: Ingest pipeline

Goal: video file in, populated index out, with measured speed.

Tasks:
1. SQLite schema from 4.5 with `config` filled in.
2. Start time resolution chain from 4.6, with unit tests for each branch.
3. Detection + ByteTrack with frame stride (default 5; make it configurable). Compute `offset_s` from frame index and fps, and absolute time from video start.
4. Best-frame selection per track, annotated snapshot saved to disk, bbox stored both in pixels and normalized.
5. Crop embedding per track, batched, with automatic out-of-memory fallback to smaller batches (log when it triggers).
6. Whole-frame embeddings about every 2 seconds into `frames`.
7. Incremental indexing: a video already indexed is skipped. Re-running must not create duplicates.
8. `scripts/ingest.py --videos DIR --manifest FILE --out INDEX_DIR`, printing per-video: duration, tracks, frames, wall time, ratio to video length, peak VRAM.
9. Tracker isolation test: ingest two different videos in one process and prove that track IDs and state from video 1 do not affect video 2 (report how you tested it).

Needs from human: at least one real video (1 to 2 minutes), and its camera name and start time. Ask and stop if missing.

Acceptance tests: real run on a real clip; row counts match what you print; snapshots exist on disk and are listed; re-run adds zero duplicate rows; stride and speed measured; unit tests pass. Open a few snapshot files only by checking they exist and their dimensions (no browser, no screenshots). The human will eyeball a sample. Put 3 file paths in MANUAL CHECK.

### PHASE 2: Query engine (search, no UI)

Goal: a Python function and CLI that turn a question into ranked results.

Tasks:
1. `parse_query(text)` returns JSON: `object_description`, `location_referent` (or null), `time_range` (absolute or relative expression), `camera` (if explicit). Use the provider interface from section 4.2 (Claude or Gemini, chosen by `PARSER_PROVIDER`) with a strict JSON-only instruction and safe parsing. The rule-based parser must also exist as the final fallback. In every test, record which provider actually answered.
   **Context bundle:** every parser call (Claude or Gemini) receives a compact context object built by code from real data: the list of camera names in the index, the saved aliases (name -> camera), the "now" timestamp from section 4.7, the index's time coverage per camera, and the last few turns of the conversation (see Phase 4 session state). The parser uses this context to understand follow-up questions (for example "and at the lobby?" or "show me that one again") and to map phrases onto known cameras and aliases. **The context must never be used to guess an unknown location referent.** If a referent is not an existing camera name and not in the saved aliases, the parser must return it as unresolved so that the clarify-once flow in Phase 3 asks the user. Include tests that prove this: a prompt where the context could tempt a guess (for example a camera named `cam_gate_2` while the user says "main gate") must still produce a clarification request.
   Run the same 15 or more test sentences (written by you, labeled as developer-written, covering relative time, explicit cameras, unknown places, and plain object queries) through every available provider and report valid-JSON rate, field correctness against your own stated expectations, and latency (median and worst of repeated runs, state the count).
2. Time parsing against "now" = latest indexed timestamp (4.7), with override. Unit tests for: "last hour", "yesterday", "between 2 and 3 PM", "this morning", plus at least 4 more cases.
3. Search: SigLIP text embedding of the object description (try prompt variants such as the bare phrase and "a photo of ..." and report which is better on real data, do not assume), cosine similarity over `tracks` and `frames`, camera/time filters applied, results merged, near-duplicate hits of the same object merged by temporal proximity on the same camera.
4. Result object: camera, absolute time range, best timestamp, score, snapshot path, source (track or frame), clip descriptor.
5. Clip cutting with ffmpeg from `video_path` using offsets, with a small padding. Output short mp4 files. Handle the case where the raw video is missing (return snapshot only and say so).
6. `scripts/query.py "question"` printing the top results.
7. Measure latency per stage: parse, text embed, search, clip cut. Report each separately, with repeated runs (state how many).

Acceptance tests: unit tests for the time parser; a real query on the Phase 1 index returns results with real camera and time; clip file exists and its duration is checked with `ffprobe`; latency table produced.

### PHASE 3: Clarify-once memory

Goal: the memory behavior worth 20% of the score, tested properly.

Tasks:
1. `aliases` store with normalization (4.8) and polygon support.
2. Flow: the query function returns `{"status": "clarify", "referent": "main gate", "options": [cameras...]}` when the referent is unknown. A separate function `save_alias(name, camera, polygon_norm=None)` stores the answer. After that, the same query resolves without asking.
3. Region filter: tracks inside the polygon only.
4. Tests (automated, in `tests/`): (a) unknown referent triggers clarification exactly once; (b) after saving, the same and similar phrasings never ask again; (c) **restart test**: run the save in one Python process, then a fresh process, and confirm no clarification and correct resolution; (d) a referent that is an existing camera name does not trigger clarification; (e) polygon filtering returns only tracks whose bbox center is inside.
5. Never ask when the information is already known. Never guess when it is unknown. Neither an LLM (Claude or Gemini) nor any conversation context may resolve an unknown referent: only the camera list and the saved `aliases` table can. Add a test that runs with a context-aware provider enabled and still gets a clarification request for an unknown referent.

Acceptance tests: all five tests above, with raw output. The restart test must use two separate processes, and you must show that.

### PHASE 4: API and chat UI

Goal: usable demo, verified without a browser.

Tasks:
1. FastAPI backend: `POST /ask` (question, optional "now" override), `POST /alias`, `GET /cameras`, `GET /snapshot/{id}`, `GET /clip/{id}`, `POST /upload` (saves a video into `footage/`, requires camera + start time or resolves via the chain), `GET /health`.
   **Session context:** `POST /ask` accepts an optional `session_id`. The server keeps the last 5 turns per session (question, parsed fields, result summary) in memory and passes them to the parser as part of the context bundle from Phase 2. Sessions are conversation memory only. They are separate from the permanent `aliases` table, and a restart may clear sessions but must never clear aliases. Test follow-ups through the API with raw output (for example a first question, then "and at the lobby?" in the same session), and test that a new session with no history does not inherit another session's context.
2. Minimal static chat UI: message list, result cards (camera, time, snapshot, clip player), a clarification card with camera choice and a simple polygon-drawing canvas on the camera snapshot.
3. Verify every endpoint through `curl` or `httpx` in tests, with raw output. Verify the restart persistence through the API as well (stop server, start, repeat query).
4. The UI itself will be checked by the human. List MANUAL CHECK steps. **Do not open a browser or take screenshots.**

Acceptance tests: endpoint tests pass; clip and snapshot endpoints return real files with correct content types; API-level clarify-once survives a server restart.

### PHASE 5: Evaluation and ablation (research contribution)

Goal: measured evidence for the 30% and 20% criteria.

The human provides `eval/queries.json`. Format:

```
[{"id":"q1","query":"red car at the gate","truth":[{"camera":"cam_gate","start":"...","end":"..."}]}, ...]
```

Tasks:
1. Build `scripts/eval.py`. Metric definitions go in `EVAL.md` before you run anything: a hit counts when the returned camera matches and the returned time falls inside (or within a stated tolerance of) a truth window. Report top-1 and top-5, and camera-only accuracy and time-localization accuracy separately.
2. Split queries into a **dev set** and a **held-out set** (the human defines the split in the file). Never tune on held-out.
3. **Baseline**: whole-frame retrieval only (frames at 1 fps, same embedder, no detection, no tracking). Define it in `EVAL.md` and implement it fairly.
4. **Ablation rows** (each is run on the same queries): baseline; + detection crops; + tracking (per-track embedding); + wider vocabulary; + whole-frame fallback; + temporal duplicate merging. Add rerank rows in Phase 6 if built.
5. Report latency and storage (index size on disk, rows) per row.
6. Produce `eval/results/*.json` and a table in the report. If there are too few queries to mean anything, say so plainly and do not over-interpret small differences.

Acceptance tests: eval runs end to end; the numbers in the report match the raw output files; held-out numbers reported separately.

### PHASE 6: Accuracy improvements (only if evidence shows the need)

Decide using Phase 5 failures, not guesses. Allowed work, in this order:
1. Look at the failure cases (list them with query id and what was returned).
2. If objects are missed: try `yolov8m-worldv2.pt`, measured for recall, speed and VRAM.
3. If the right answer is in the top 10 but not top 1: add vision-language rerank of the top 10 snapshots with yes/no plus confidence, using `judge_image` from the provider interface (Gemini or Claude, chosen by `RERANK_PROVIDER`), as an option that can be switched on and off. Send the images in one batched or parallel request where the provider allows it, and measure whether that helps latency. Measure accuracy and added latency separately for each provider that has a key, and report the tradeoff honestly, including any rate-limit effects. Remember that rerank sends snapshots to the cloud: state this in the report.
4. If wrong attribute (such as color) results dominate: try prompt variants or attribute-specific checks and measure.
Each change gets its own ablation row. Keep only what helps on the dev set, and confirm on held-out.

### PHASE 7: Stretch goals and hardening (only after Phases 0 to 6 pass)

Order: (a) unified `watch.py` for both folders, plus `record.py` for RTSP phone streams (test with a recorded file served as a stream if no phone is available, and label it as such); (b) cheap cross-camera re-identification using track embeddings within a time window, with a per-path report ("cam A time -> cam B time"), measured honestly with a small test; (c) standing queries or alerts only if time remains; (d) privacy notes (what runs locally, what calls the cloud: the Claude or Gemini parser and the rerank are cloud calls, so state that plainly, including that rerank sends snapshot images to the chosen provider). Also report how the system behaves with `PARSER_PROVIDER=rules` and `RERANK_PROVIDER=off`, which is the fully local mode, and measure its accuracy and latency honestly.
Also: README polish, a demo script, a draft of the short write-up with real numbers only.

---

## 8. STARTING INSTRUCTIONS

1. Read this whole prompt. Do not write code yet.
2. Create `RULES.md`, `PLAN.md`, `TASKS.md`, `ARCHITECTURE.md`, `DECISIONS.md`, `EVAL.md` first (docs only), then show the human a short summary of your understanding and any questions or problems you noticed in this prompt.
3. Then begin **Phase 0** and stop at its end with `REPORTS/PHASE_0_REPORT.md`.
4. Never proceed to the next phase without the exact message "APPROVED PHASE N" from the human.

Reminder of the three rules that matter most: **no fake results, no browser or screenshots, stop and report at the end of every phase.**
