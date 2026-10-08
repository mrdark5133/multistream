import sys
import os
import json
import time
import hashlib
import sqlite3
import datetime
import statistics
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

# Add project root
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.query.search import SearchEngine, SearchResult
from src.query.parser import QueryParser
from src.index.db import blob_to_emb
import numpy as np


DELTA_SECONDS = 5.0  # Temporal tolerance margin per EVAL.md


def is_hit(result: SearchResult, truth_list: List[Dict[str, Any]], delta_s: float = DELTA_SECONDS) -> Tuple[bool, float]:
    """
    Check if a result is a hit against ground truth list.
    Returns (is_hit, abs_time_error_seconds).
    """
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
            t_start = datetime.datetime.fromisoformat(t["start"]) - datetime.timedelta(seconds=delta_s)
            t_end = datetime.datetime.fromisoformat(t["end"]) + datetime.timedelta(seconds=delta_s)

            # Error from raw ground truth interval (without delta)
            raw_start = datetime.datetime.fromisoformat(t["start"])
            raw_end = datetime.datetime.fromisoformat(t["end"])

            if r_dt < raw_start:
                err = (raw_start - r_dt).total_seconds()
            elif r_dt > raw_end:
                err = (r_dt - raw_end).total_seconds()
            else:
                err = 0.0

            best_error = min(best_error, err)

            if t_start <= r_dt <= t_end:
                return True, err

    # If camera matched but outside time tolerance
    if matched_cam:
        return False, best_error

    return False, 999.0


def evaluate_queries(
    queries: List[Dict[str, Any]],
    engine: SearchEngine,
    mode: str = "full",  # 'full', 'whole_frame', 'no_tracking', 'bare_prompt', 'photo_prompt'
    top_k: int = 5
) -> Dict[str, Any]:
    """
    Run evaluation harness across queries and compute metrics.
    """
    mrr_list = []
    top1_hits = 0
    top5_hits = 0
    time_errors = []
    latencies = []

    for q_item in queries:
        q_text = q_item["query"]
        truth = q_item["truth"]

        t0 = time.perf_counter()
        
        # 1. Parse
        now_ref = engine.get_latest_timestamp()
        parsed = engine.parser.parse(q_text, now_ref=now_ref)

        # Prompt customization for ablations
        if mode == "bare_prompt":
            # Strip "a " prefix
            prompt = parsed.object_prompt
            if prompt.startswith("a "):
                prompt = prompt[2:]
            elif prompt.startswith("an "):
                prompt = prompt[3:]
        elif mode == "photo_prompt":
            prompt = f"a photo of {parsed.object_prompt}"
        else:
            prompt = parsed.object_prompt

        query_emb = engine.embedder.embed_text([prompt])[0]

        # 2. Retrieve candidates according to mode
        candidates: List[SearchResult] = []

        if mode == "whole_frame":
            # Whole-frame baseline: query frames table only
            frame_sql = "SELECT id, video, camera, t_abs, offset_s, snapshot, emb FROM frames WHERE 1=1"
            params = []
            if parsed.location_status == "RESOLVED" and parsed.resolved_camera:
                frame_sql += " AND (LOWER(camera) LIKE ? OR LOWER(camera) LIKE ?)"
                params.extend([f"%{parsed.resolved_camera.lower()}%", f"%cam_{parsed.resolved_camera.lower()}%"])
            rows = engine.conn.execute(frame_sql, params).fetchall()
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
            # Query tracks table
            track_sql = "SELECT id, video, camera, track_id, label, t_start, t_end, t_best, offset_start, offset_end, offset_best, bbox_px, bbox_norm, snapshot, emb FROM tracks WHERE 1=1"
            params = []
            if parsed.location_status == "RESOLVED" and parsed.resolved_camera:
                track_sql += " AND (LOWER(camera) LIKE ? OR LOWER(camera) LIKE ?)"
                params.extend([f"%{parsed.resolved_camera.lower()}%", f"%cam_{parsed.resolved_camera.lower()}%"])
            
            rows = engine.conn.execute(track_sql, params).fetchall()
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

            # Add whole-frame fallback
            frame_sql = "SELECT id, video, camera, t_abs, offset_s, snapshot, emb FROM frames WHERE 1=1"
            frame_params = []
            if parsed.location_status == "RESOLVED" and parsed.resolved_camera:
                frame_sql += " AND (LOWER(camera) LIKE ? OR LOWER(camera) LIKE ?)"
                frame_params.extend([f"%{parsed.resolved_camera.lower()}%", f"%cam_{parsed.resolved_camera.lower()}%"])
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
                # Skip temporal deduplication: take top_k raw hits directly
                results = candidates[:top_k]
            else:
                # Full deduplication
                deduped = []
                for cand in candidates:
                    is_dup = False
                    for existing in deduped:
                        if existing.camera == cand.camera:
                            try:
                                d1 = datetime.datetime.fromisoformat(existing.timestamp)
                                d2 = datetime.datetime.fromisoformat(cand.timestamp)
                                if abs((d1 - d2).total_seconds()) <= 5.0:
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

        # 3. Compute rank metrics
        hit_rank = 0
        min_err = None

        for rank_idx, r in enumerate(results[:top_k], start=1):
            hit, err = is_hit(r, truth)
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
        "mrr": round(float(np.mean(mrr_list)), 4) if mrr_list else 0.0,
        "recall_1": round(top1_hits / n_q, 4) if n_q > 0 else 0.0,
        "recall_5": round(top5_hits / n_q, 4) if n_q > 0 else 0.0,
        "mean_time_error_s": round(float(np.mean(time_errors)), 2) if time_errors else 0.0,
        "latency_median_ms": round(float(statistics.median(latencies)), 2) if latencies else 0.0,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2) if latencies else 0.0
    }


def main():
    queries_file = ROOT_DIR / "eval" / "queries.json"
    results_dir = ROOT_DIR / "eval" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    if not queries_file.exists():
        print(f"Error: {queries_file} not found.")
        sys.exit(1)

    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Compute SHA256 of eval queries
    file_bytes = open(queries_file, "rb").read()
    sha256 = hashlib.sha256(file_bytes).hexdigest()
    print("=" * 90)
    print(f"EVALUATION HARNESS (EVAL.md)")
    print(f"Queries File: {queries_file}")
    print(f"Queries SHA256: {sha256}")
    print("=" * 90)

    dev_queries = [q for q in data if q.get("split") == "dev"]
    heldout_queries = [q for q in data if q.get("split") == "held-out"]
    print(f"Loaded {len(dev_queries)} dev queries and {len(heldout_queries)} held-out queries.")

    db_path = ROOT_DIR / "index_base" / "index.db"
    engine = SearchEngine(db_path=db_path, device="cuda:0")

    # Measure storage footprint
    db_size_mb = db_path.stat().st_size / (1024 * 1024)
    snaps_dir = ROOT_DIR / "index_base" / "snapshots"
    snaps_size_mb = sum(f.stat().st_size for f in snaps_dir.rglob("*") if f.is_file()) / (1024 * 1024)
    total_index_mb = db_size_mb + snaps_size_mb

    # Index throughput (from Phase 1 measurement: 64.7s video in 19.34s)
    throughput_rtf = 64.7 / 19.34

    # Run Ablation Matrix
    modes = [
        ("Full Pipeline (Ours)", "full"),
        ("Whole-Frame Baseline", "whole_frame"),
        ("No-Tracking Ablation", "no_tracking"),
        ("Prompt Variant: Bare", "bare_prompt"),
        ("Prompt Variant: 'a photo of...'", "photo_prompt")
    ]

    ablation_results = []

    print("\n--- Running Ablation Matrix on Dev Split ---")
    for name, mode in modes:
        dev_res = evaluate_queries(dev_queries, engine, mode=mode, top_k=5)
        # Run on held-out only for final report
        heldout_res = evaluate_queries(heldout_queries, engine, mode=mode, top_k=5)
        
        row = {
            "configuration": name,
            "mode": mode,
            "dev_mrr": dev_res["mrr"],
            "dev_r1": dev_res["recall_1"],
            "dev_r5": dev_res["recall_5"],
            "dev_time_err": dev_res["mean_time_error_s"],
            "heldout_mrr": heldout_res["mrr"],
            "heldout_r1": heldout_res["recall_1"],
            "heldout_r5": heldout_res["recall_5"],
            "heldout_time_err": heldout_res["mean_time_error_s"],
            "latency_median_ms": dev_res["latency_median_ms"],
            "latency_p95_ms": dev_res["latency_p95_ms"]
        }
        ablation_results.append(row)
        print(f"Config: {name:<32} | Dev R@1={dev_res['recall_1']*100:5.1f}% | Dev R@5={dev_res['recall_5']*100:5.1f}% | Held-Out R@1={heldout_res['recall_1']*100:5.1f}% | Held-Out R@5={heldout_res['recall_5']*100:5.1f}% | Latency={dev_res['latency_median_ms']:5.1f}ms")

    # Output full JSON
    output_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "eval_queries_sha256": sha256,
        "storage": {
            "database_mb": round(db_size_mb, 2),
            "snapshots_mb": round(snaps_size_mb, 2),
            "total_index_mb": round(total_index_mb, 2),
            "indexing_throughput_rtf": round(throughput_rtf, 2)
        },
        "ablations": ablation_results
    }

    out_json_path = results_dir / "eval_summary.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved machine-readable evaluation results to {out_json_path}")

    # Format Markdown Table
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS (Dev and Held-Out Splits)")
    print("=" * 90)
    print(f"| Configuration | Dev MRR | Dev R@1 | Dev R@5 | Held-Out MRR | Held-Out R@1 | Held-Out R@5 | Median Latency |")
    print(f"|---|---|---|---|---|---|---|---|")
    for r in ablation_results:
        print(f"| {r['configuration']:<28} | {r['dev_mrr']:.4f} | {r['dev_r1']*100:5.1f}% | {r['dev_r5']*100:5.1f}% | {r['heldout_mrr']:.4f} | {r['heldout_r1']*100:5.1f}% | {r['heldout_r5']*100:5.1f}% | {r['latency_median_ms']:6.1f} ms |")
    print("=" * 90)

    print("\nSTORAGE & EFFICIENCY FOOTPRINT:")
    print(f"- SQLite Index Database: {db_size_mb:.2f} MB")
    print(f"- Snapshot Evidence Cache: {snaps_size_mb:.2f} MB")
    print(f"- Total Index Footprint: {total_index_mb:.2f} MB")
    print(f"- Ingest Throughput: {throughput_rtf:.2f}x Real-Time ({throughput_rtf:.2f} video seconds processed per second)")


if __name__ == "__main__":
    main()
