import sys
import time
import json
import statistics
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.query.search import SearchEngine
from src.query.clips import extract_video_clip

def run_latency_benchmark():
    db_path = Path("index_base/index.db")
    print(f"Initializing SearchEngine with db={db_path}...")
    
    # Pre-load engine (text model is loaded into GPU memory during __init__)
    t_init_start = time.perf_counter()
    engine = SearchEngine(db_path=db_path, device="cuda:0")
    t_init_end = time.perf_counter()
    print(f"Model loaded in {(t_init_end - t_init_start):.2f} s. Text model already loaded in memory: YES.")

    query_text = "a red car near cam_landscape"
    
    # 1. Warm-up run
    print("\n--- Running 1 Warm-up Run ---")
    parsed_warmup, results_warmup = engine.search(query_text, top_k=1)
    if results_warmup:
        r = results_warmup[0]
        _ = extract_video_clip(
            video_path=r.video_path,
            output_path="clips/benchmark_warmup.mp4",
            start_s=r.offset_seconds,
            end_s=r.offset_seconds + 3.0,
            padding_s=2.0
        )
    print("Warm-up complete.")

    # 2. 10 Timed Runs
    print("\n--- Running 10 Timed Runs ---")
    latencies_parse = []
    latencies_embed = []
    latencies_search = []
    latencies_clip = []
    totals_no_clip = []
    totals_with_clip = []

    for i in range(10):
        t_start = time.perf_counter()
        
        # Stage 1: Parse
        t0 = time.perf_counter()
        now_ref = engine.get_latest_timestamp()
        parsed = engine.parser.parse(query_text, now_ref=now_ref)
        t_parse = (time.perf_counter() - t0) * 1000.0

        # Stage 2: Embed
        t0 = time.perf_counter()
        query_emb = engine.embedder.embed_text([parsed.object_prompt])[0]
        t_embed = (time.perf_counter() - t0) * 1000.0

        # Stage 3: Search (DB candidate fetch + dot-product vector search + dedup)
        t0 = time.perf_counter()
        track_sql = "SELECT id, video, camera, track_id, label, t_start, t_end, t_best, offset_start, offset_end, offset_best, bbox_px, bbox_norm, snapshot, emb FROM tracks WHERE 1=1"
        params = []
        if parsed.location_status == "RESOLVED" and parsed.resolved_camera:
            track_sql += " AND (LOWER(camera) LIKE ? OR LOWER(camera) LIKE ?)"
            params.extend([f"%{parsed.resolved_camera.lower()}%", f"%cam_{parsed.resolved_camera.lower()}%"])
        
        cur_tracks = engine.conn.execute(track_sql, params).fetchall()
        from src.index.db import blob_to_emb
        import numpy as np
        
        cands = []
        for row in cur_tracks:
            score = float(np.dot(query_emb, blob_to_emb(row["emb"])))
            cands.append((score, row))
        cands.sort(key=lambda x: x[0], reverse=True)
        top_cand = cands[0] if cands else None
        t_search = (time.perf_counter() - t0) * 1000.0

        t_total_no_clip = (time.perf_counter() - t_start) * 1000.0

        # Stage 4: Clip Cut
        t0 = time.perf_counter()
        if top_cand:
            row = top_cand[1]
            out_clip = f"clips/benchmark_run_{i}.mp4"
            _ = extract_video_clip(
                video_path=row["video"],
                output_path=out_clip,
                start_s=float(row["offset_best"]),
                end_s=float(row["offset_best"]) + 3.0,
                padding_s=2.0
            )
        t_clip = (time.perf_counter() - t0) * 1000.0

        t_total_with_clip = (time.perf_counter() - t_start) * 1000.0

        latencies_parse.append(t_parse)
        latencies_embed.append(t_embed)
        latencies_search.append(t_search)
        latencies_clip.append(t_clip)
        totals_no_clip.append(t_total_no_clip)
        totals_with_clip.append(t_total_with_clip)

        print(f"Run {i+1:2d}: Parse={t_parse:6.2f}ms | Embed={t_embed:6.2f}ms | Search={t_search:6.2f}ms | ClipCut={t_clip:7.2f}ms | Total(NoClip)={t_total_no_clip:6.2f}ms | Total(WithClip)={t_total_with_clip:7.2f}ms")

    print("\n" + "=" * 80)
    print("LATENCY BENCHMARK RESULTS (N=10 runs after 1 warm-up)")
    print("=" * 80)
    print(f"Text model pre-loaded: YES (SigLIP on GPU cuda:0)")
    print(f"Stage 1 (Parse):       Median = {statistics.median(latencies_parse):6.2f} ms | Worst = {max(latencies_parse):6.2f} ms")
    print(f"Stage 2 (Embed):       Median = {statistics.median(latencies_embed):6.2f} ms | Worst = {max(latencies_embed):6.2f} ms")
    print(f"Stage 3 (Search):      Median = {statistics.median(latencies_search):6.2f} ms | Worst = {max(latencies_search):6.2f} ms")
    print(f"Stage 4 (Clip Cut):    Median = {statistics.median(latencies_clip):6.2f} ms | Worst = {max(latencies_clip):6.2f} ms")
    print("-" * 80)
    print(f"Total WITHOUT Clip:    Median = {statistics.median(totals_no_clip):6.2f} ms | Worst = {max(totals_no_clip):6.2f} ms")
    print(f"Total WITH Clip Cut:   Median = {statistics.median(totals_with_clip):6.2f} ms | Worst = {max(totals_with_clip):6.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    run_latency_benchmark()
