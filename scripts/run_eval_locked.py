import sys
import json
import time
import datetime
import statistics
import hashlib
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.query.search import SearchEngine, SearchResult
from src.index.db import blob_to_emb

# 1. Hashes
locked_bytes = subprocess.check_output(['git', 'show', '3e942db:eval/queries.json'])
locked_sha256 = hashlib.sha256(locked_bytes).hexdigest()
locked_queries = json.loads(locked_bytes.decode('utf-8'))

cur_bytes = open(ROOT_DIR / 'eval' / 'queries.json', 'rb').read()
cur_sha256 = hashlib.sha256(cur_bytes).hexdigest()

print("=" * 100)
print(f"LOCKED QUERIES SHA256 (at commit 3e942db): {locked_sha256}")
print(f"CURRENT QUERIES SHA256 (on disk):           {cur_sha256}")
print("=" * 100)

SYNONYM_MAP = {
    "car": {"car", "van", "suv"},
    "van": {"car", "van", "suv"},
    "suv": {"car", "van", "suv"},
    "truck": {"truck", "bus"},
    "bus": {"bus", "truck"},
    "person": {"person", "pedestrian"},
    "pedestrian": {"person", "pedestrian"},
    "motorcycle": {"motorcycle", "motorbike"},
    "motorbike": {"motorcycle", "motorbike"},
    "umbrella": {"umbrella"},
    "cat": {"cat", "animal"},
    "utility cart": {"utility cart"},
}

def label_matches(result_label: str, truth_label: str) -> bool:
    if result_label == "whole_frame":
        return True
    r = result_label.lower().strip()
    t = truth_label.lower().strip()
    if r == t:
        return True
    return r in SYNONYM_MAP.get(t, {t})

def is_hit_phase5(result: SearchResult, truth_list: List[Dict[str, Any]], delta_s: float = 5.0) -> Tuple[bool, float]:
    r_cam = result.camera.lower()
    try:
        r_dt = datetime.datetime.fromisoformat(result.timestamp)
    except Exception:
        return False, 999.0

    best_error = 999.0
    matched_cam = False

    for t in truth_list:
        t_cam = t["camera"].lower()
        if r_cam == t_cam or r_cam == f"cam_{t_cam}" or t_cam == f"cam_{r_cam}":
            matched_cam = True
            raw_start = datetime.datetime.fromisoformat(t["start"])
            raw_end = datetime.datetime.fromisoformat(t["end"])
            t_start = raw_start - datetime.timedelta(seconds=delta_s)
            t_end = raw_end + datetime.timedelta(seconds=delta_s)

            if r_dt < raw_start:
                err = (raw_start - r_dt).total_seconds()
            elif r_dt > raw_end:
                err = (r_dt - raw_end).total_seconds()
            else:
                err = 0.0

            best_error = min(best_error, err)
            if t_start <= r_dt <= t_end:
                return True, err
    if matched_cam:
        return False, best_error
    return False, 999.0


def is_hit_phase6(result: SearchResult, truth_list: List[Dict[str, Any]], delta_s: float = 3.0) -> Tuple[bool, float]:
    r_cam = result.camera.lower()
    try:
        r_dt = datetime.datetime.fromisoformat(result.timestamp)
    except Exception:
        return False, 999.0

    best_error = 999.0
    matched_cam = False

    for t in truth_list:
        t_cam = t["camera"].lower()
        if r_cam == t_cam or r_cam == f"cam_{t_cam}" or t_cam == f"cam_{r_cam}":
            matched_cam = True
            t_label = t.get("label", "").lower().strip()
            if not label_matches(result.label, t_label):
                continue

            raw_start = datetime.datetime.fromisoformat(t["start"])
            raw_end = datetime.datetime.fromisoformat(t["end"])
            t_start = raw_start - datetime.timedelta(seconds=delta_s)
            t_end = raw_end + datetime.timedelta(seconds=delta_s)

            if r_dt < raw_start:
                err = (raw_start - r_dt).total_seconds()
            elif r_dt > raw_end:
                err = (r_dt - raw_end).total_seconds()
            else:
                err = 0.0

            best_error = min(best_error, err)
            if t_start <= r_dt <= t_end:
                return True, err
    if matched_cam:
        return False, best_error
    return False, 999.0


def eval_suite(queries: List[Dict[str, Any]], engine: SearchEngine, mode: str, hit_fn, delta_s: float, top_k: int = 5):
    mrr_list = []
    top1_hits = 0
    top5_hits = 0
    time_errors = []
    latencies = []

    for q_item in queries:
        q_text = q_item["query"]
        truth = q_item["truth"]

        t0 = time.perf_counter()
        now_ref = engine.get_latest_timestamp()
        parsed = engine.parser.parse(q_text, now_ref=now_ref)

        if mode == "bare_prompt":
            prompt = parsed.object_prompt
            if prompt.startswith("a "): prompt = prompt[2:]
            elif prompt.startswith("an "): prompt = prompt[3:]
        elif mode == "photo_prompt":
            prompt = f"a photo of {parsed.object_prompt}"
        else:
            prompt = parsed.object_prompt

        query_emb = engine.embedder.embed_text([prompt])[0]

        if mode == "whole_frame":
            frame_sql = "SELECT id, video, camera, t_abs, offset_s, snapshot, emb FROM frames WHERE 1=1"
            params = []
            if parsed.location_status == "RESOLVED" and parsed.resolved_camera:
                cam_norm = parsed.resolved_camera.lower()
                cam_bare = cam_norm.replace("cam_", "")
                cam_with = f"cam_{cam_bare}"
                frame_sql += " AND LOWER(camera) IN (?, ?, ?)"
                params.extend([cam_norm, cam_bare, cam_with])
            rows = engine.conn.execute(frame_sql, params).fetchall()
            candidates = []
            for r in rows:
                score = float(np.dot(query_emb, blob_to_emb(r["emb"])))
                candidates.append(SearchResult(
                    result_id=r["id"],
                    result_type="frame",
                    camera=r["camera"],
                    timestamp=r["t_abs"],
                    offset_seconds=float(r["offset_s"]),
                    score=score,
                    label="whole_frame",
                    video_path=r["video"],
                    snapshot_path=r["snapshot"]
                ))
            candidates.sort(key=lambda x: x.score, reverse=True)
            results = candidates[:top_k]
        else:
            track_sql = "SELECT id, video, camera, track_id, label, t_start, t_end, t_best, offset_start, offset_end, offset_best, bbox_px, bbox_norm, snapshot, emb FROM tracks WHERE 1=1"
            params = []
            if parsed.location_status == "RESOLVED" and parsed.resolved_camera:
                cam_norm = parsed.resolved_camera.lower()
                cam_bare = cam_norm.replace("cam_", "")
                cam_with = f"cam_{cam_bare}"
                track_sql += " AND LOWER(camera) IN (?, ?, ?)"
                params.extend([cam_norm, cam_bare, cam_with])
            rows = engine.conn.execute(track_sql, params).fetchall()
            candidates = []
            for r in rows:
                score = float(np.dot(query_emb, blob_to_emb(r["emb"])))
                candidates.append(SearchResult(
                    result_id=r["id"],
                    result_type="track",
                    camera=r["camera"],
                    timestamp=r["t_best"],
                    offset_seconds=float(r["offset_best"]),
                    score=score,
                    label=r["label"],
                    video_path=r["video"],
                    snapshot_path=r["snapshot"]
                ))
            frame_sql = "SELECT id, video, camera, t_abs, offset_s, snapshot, emb FROM frames WHERE 1=1"
            frame_params = []
            if parsed.location_status == "RESOLVED" and parsed.resolved_camera:
                cam_norm = parsed.resolved_camera.lower()
                cam_bare = cam_norm.replace("cam_", "")
                cam_with = f"cam_{cam_bare}"
                frame_sql += " AND LOWER(camera) IN (?, ?, ?)"
                frame_params.extend([cam_norm, cam_bare, cam_with])
            for r in engine.conn.execute(frame_sql, frame_params).fetchall():
                score = float(np.dot(query_emb, blob_to_emb(r["emb"])))
                candidates.append(SearchResult(
                    result_id=r["id"],
                    result_type="frame",
                    camera=r["camera"],
                    timestamp=r["t_abs"],
                    offset_seconds=float(r["offset_s"]),
                    score=score,
                    label="whole_frame",
                    video_path=r["video"],
                    snapshot_path=r["snapshot"]
                ))
            candidates.sort(key=lambda x: x.score, reverse=True)
            if mode == "no_tracking":
                results = candidates[:top_k]
            else:
                deduped = []
                for cand in candidates:
                    is_dup = False
                    for existing in deduped:
                        if existing.camera == cand.camera:
                            try:
                                d1 = datetime.datetime.fromisoformat(existing.timestamp)
                                d2 = datetime.datetime.fromisoformat(cand.timestamp)
                                if abs((d1 - d2).total_seconds()) <= 3.0:
                                    is_dup = True
                                    break
                            except Exception:
                                pass
                    if not is_dup:
                        deduped.append(cand)
                    if len(deduped) >= top_k:
                        break
                results = deduped

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        hit_rank = 0
        min_err = None
        for rank_idx, r in enumerate(results[:top_k], start=1):
            hit, err = hit_fn(r, truth, delta_s=delta_s)
            if hit and hit_rank == 0:
                hit_rank = rank_idx
                min_err = err

        if hit_rank == 1:
            top1_hits += 1
            top5_hits += 1
            mrr_list.append(1.0)
            time_errors.append(min_err if min_err is not None else 0.0)
        elif 1 < hit_rank <= 5:
            top5_hits += 1
            mrr_list.append(1.0 / hit_rank)
            time_errors.append(min_err if min_err is not None else 0.0)
        else:
            mrr_list.append(0.0)

    n_q = len(queries)
    return {
        "num_queries": n_q,
        "mrr": round(float(np.mean(mrr_list)), 4) if n_q else 0.0,
        "recall_1": round(top1_hits / n_q, 4) if n_q else 0.0,
        "recall_5": round(top5_hits / n_q, 4) if n_q else 0.0,
        "mean_time_error_s": round(float(np.mean(time_errors)), 2) if time_errors else 0.0,
        "latency_median_ms": round(float(statistics.median(latencies)), 2) if latencies else 0.0,
    }


def run_eval_matrix(db_path: Path, hit_fn, delta_s: float, label: str):
    print("\n" + "=" * 100)
    print(f"DATABASE: {db_path.name} | CRITERIA: {label}")
    print("=" * 100)
    engine = SearchEngine(db_path=db_path, device="cuda:0")

    dev_queries = [q for q in locked_queries if q.get("split") == "dev"]
    heldout_queries = [q for q in locked_queries if q.get("split") == "held-out"]

    modes = [
        ("Full Pipeline (Ours)", "full"),
        ("Whole-Frame Baseline", "whole_frame"),
        ("No-Tracking Ablation", "no_tracking"),
        ("Prompt Variant: Bare", "bare_prompt"),
        ("Prompt Variant: 'a photo of...'", "photo_prompt")
    ]

    print(f"| {'Configuration':<32} | Dev MRR | Dev R@1 | Dev R@5 | Held-Out MRR | Held-Out R@1 | Held-Out R@5 | Median Lat |")
    print(f"|{'-'*34}|{'-'*9}|{'-'*9}|{'-'*9}|{'-'*14}|{'-'*14}|{'-'*14}|{'-'*12}|")
    for name, mode in modes:
        dev_res = eval_suite(dev_queries, engine, mode=mode, hit_fn=hit_fn, delta_s=delta_s)
        heldout_res = eval_suite(heldout_queries, engine, mode=mode, hit_fn=hit_fn, delta_s=delta_s)
        print(f"| {name:<32} | {dev_res['mrr']:.4f}  | {dev_res['recall_1']*100:5.1f}%  | {dev_res['recall_5']*100:5.1f}%  | {heldout_res['mrr']:.4f}       | {heldout_res['recall_1']*100:5.1f}%        | {heldout_res['recall_5']*100:5.1f}%        | {dev_res['latency_median_ms']:6.1f} ms |")


if __name__ == "__main__":
    fresh_db = ROOT_DIR / "index_fresh" / "index.db"
    phase1b_db = ROOT_DIR / "index_phase1b" / "index.db"

    # Evaluation on index_fresh with Original Phase 5 criteria (Camera + Time +-5s, no label match)
    run_eval_matrix(fresh_db, is_hit_phase5, delta_s=5.0, label="Phase 5 Original (Camera+Time +-5s, Locked Queries)")

    # Evaluation on index_fresh with Strict Label Matching (Camera + Time +-3s + Label Match)
    run_eval_matrix(fresh_db, is_hit_phase6, delta_s=3.0, label="Phase 6 Strict Criteria (Camera+Time +-3s + Label Match, Locked Queries)")
