import os
import sys
import cv2
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
from src.ingest.detector import HybridDetector, COCO_KEEP_CLASSES, get_object_group
from src.ingest.detect_track import (
    load_yolo_world,
    process_video_tracks,
    process_video_tracks_phase1b,
    TrackSummary
)
from src.ingest.track_engine import (
    create_tracker,
    pad_to_square,
    compute_crop_quality,
    filter_tracks,
    stitch_tracks
)
from src.utils.lab_color import extract_track_colors, extract_lab_kmeans_colors
from src.ingest.embed import SigLIPEmbedder
from src.ingest.pipeline import IngestPipeline
from src.index.db import get_db

DEV_CLIPS = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4",
    "footage/test_landscape.mp4",
    "footage/test_landscape2.mp4"
]

HAND_COUNTED_REAL_OBJECTS = {
    "footage/test_video01.mp4": 1,   # 1 person walking across
    "footage/test_video02.mp4": 1,   # 1 animal in frame
    "footage/test_video03.mp4": 3,   # 1 motorcycle + 2 riders
    "footage/test_landscape.mp4": 8, # 8 moving vehicles on road
    "footage/test_landscape2.mp4": 6 # 6 moving vehicles on road
}


def run_experiment_1_detector():
    """
    1. Detector: COCO-trained YOLO11 vs YOLO-World alone on dev clips.
    Reports raw detections per class for both setups.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: DETECTOR EVALUATION (YOLO-World Alone vs Hybrid YOLO11+World)")
    print("=" * 80)

    # Setup A: YOLO-World Alone
    yolo_world, world_classes = load_yolo_world(
        weights_path="yolov8s-worldv2.pt",
        vocab_path="config/vocab.yaml",
        device="cuda:0"
    )

    # Setup B: Hybrid Detector
    hybrid = HybridDetector(
        coco_weights="yolo11s.pt",
        world_weights="yolov8s-worldv2.pt",
        vocab_path="config/vocab.yaml",
        device="cuda:0"
    )

    setup_a_counts = Counter()
    setup_b_counts = Counter()
    setup_a_per_clip = {}
    setup_b_per_clip = {}

    for clip_path in DEV_CLIPS:
        cap = cv2.VideoCapture(clip_path)
        frame_idx = 0
        clip_a_counter = Counter()
        clip_b_counter = Counter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 5 == 0:  # sample every 5 frames for detector benchmark
                # Setup A
                res_a = yolo_world.predict(frame, conf=0.25, imgsz=640, verbose=False)[0]
                if res_a.boxes is not None:
                    for c in res_a.boxes.cls.tolist():
                        cls_idx = int(c)
                        lbl = world_classes[cls_idx] if cls_idx < len(world_classes) else f"cls_{cls_idx}"
                        clip_a_counter[lbl] += 1
                        setup_a_counts[lbl] += 1

                # Setup B
                dets_b = hybrid.detect(frame, imgsz=640)
                for d in dets_b:
                    lbl = d["label"]
                    clip_b_counter[lbl] += 1
                    setup_b_counts[lbl] += 1

            frame_idx += 1
        cap.release()
        setup_a_per_clip[clip_path] = clip_a_counter
        setup_b_per_clip[clip_path] = clip_b_counter

    print("\n--- Setup A: YOLO-World Alone Detections Per Class ---")
    for lbl, count in sorted(setup_a_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:22s}: {count:4d}")

    print("\n--- Setup B: Hybrid YOLO11 (COCO) + YOLO-World Detections Per Class ---")
    for lbl, count in sorted(setup_b_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lbl:22s}: {count:4d}")

    print("\nPer-Clip Breakdown:")
    for clip in DEV_CLIPS:
        print(f"\nClip: {clip}")
        print("  Setup A (World Alone):", dict(setup_a_per_clip[clip]))
        print("  Setup B (Hybrid):     ", dict(setup_b_per_clip[clip]))

    return setup_a_counts, setup_b_counts


def run_experiment_2_tracker_speed_and_tracks():
    """
    2. Tracker: lower stride to 2-3 and report speed;
    evaluate bytetrack_tuned vs botsort;
    Report tracks per hand-counted real object, before and after.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: TRACKER EVALUATION & STRIDE BENCHMARK")
    print("=" * 80)

    hybrid = HybridDetector(device="cuda:0")

    # Stride & speed benchmark on test_landscape.mp4 (452 frames, 15s)
    bench_clip = "footage/test_landscape.mp4"
    strides_to_test = [5, 3, 2]
    stride_results = {}

    print(f"\nMeasuring FPS & Latency on {bench_clip} (452 frames) across strides:")
    for s in strides_to_test:
        t0 = time.perf_counter()
        cap = cv2.VideoCapture(bench_clip)
        frames_proc = 0
        f_idx = 0
        tracker = create_tracker("bytetrack_tuned")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            if f_idx % s == 0:
                dets = hybrid.detect(frame, imgsz=640)
                from src.ingest.track_engine import run_tracker_on_detections
                _ = run_tracker_on_detections(tracker, dets, frame, w, h)
                frames_proc += 1
            f_idx += 1
        cap.release()
        elapsed = time.perf_counter() - t0
        fps = 452.0 / elapsed
        stride_results[s] = {"elapsed_s": round(elapsed, 2), "fps": round(fps, 2), "step_frames": frames_proc}
        print(f"  Stride {s}: Elapsed = {elapsed:.2f}s | Speed = {fps:.1f} FPS ({frames_proc} tracked frames)")

    # Test BoT-SORT speed at stride 2
    t0 = time.perf_counter()
    cap = cv2.VideoCapture(bench_clip)
    tracker_bot = create_tracker("botsort")
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if f_idx % 2 == 0:
            dets = hybrid.detect(frame, imgsz=640)
            from src.ingest.track_engine import run_tracker_on_detections
            _ = run_tracker_on_detections(tracker_bot, dets, frame, w, h)
        f_idx += 1
    cap.release()
    bot_elapsed = time.perf_counter() - t0
    bot_fps = 452.0 / bot_elapsed
    print(f"  BoT-SORT (Stride 2): Elapsed = {bot_elapsed:.2f}s | Speed = {bot_fps:.1f} FPS")

    # Measure Tracks per Hand-Counted Real Object (Before vs After)
    print("\n--- Tracks Per Hand-Counted Real Object (Before vs After) ---")
    print(f"{'Video Clip':30s} | {'Real Objs':9s} | {'Before (Stride 5)':18s} | {'After (Tuned Stride 2)':22s} | {'Tracks/Obj Before':17s} | {'Tracks/Obj After':16s}")
    print("-" * 125)

    embedder = SigLIPEmbedder(device="cuda:0")
    tracker_comparison = {}

    for clip in DEV_CLIPS:
        real_n = HAND_COUNTED_REAL_OBJECTS[clip]
        
        # Before: Stride 5, untuned ByteTrack, no strict filter/stitching
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()

        # Run Phase 1b pipeline with Stride 2, Tuned ByteTrack, Filter & Stitching
        tracks_after, _, dropped_counts, merges = process_video_tracks_phase1b(
            video_path=clip,
            detector=hybrid,
            start_time_iso="2026-10-08T18:00:00",
            fps=fps,
            duration_s=dur,
            stride=2,
            tracker_name="bytetrack_tuned",
            embedder=embedder
        )

        # Query existing tracks from index_base (Before)
        conn = get_db("index_base/index.db")
        cur = conn.execute("SELECT count(*) FROM tracks WHERE video = ? OR video = ?;", (clip, clip.replace("\\", "/")))
        n_before = cur.fetchone()[0]
        n_after = len(tracks_after)

        ratio_before = n_before / real_n
        ratio_after = n_after / real_n

        tracker_comparison[clip] = {
            "real_n": real_n,
            "n_before": n_before,
            "n_after": n_after,
            "dropped": dropped_counts,
            "merges": len(merges)
        }

        print(f"{clip:30s} | {real_n:9d} | {n_before:18d} | {n_after:22d} | {ratio_before:17.2f} | {ratio_after:16.2f}")

    return stride_results, bot_fps, tracker_comparison


def run_experiment_3_and_4_track_filter_and_stitching():
    """
    3. Track Filter & Quality: Report dropped tracks and reasons.
    4. Stitching: Report 10 merged pairs with gap, center distance, cosine similarity.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3 & 4: TRACK FILTERING & STITCHING EVALUATION")
    print("=" * 80)

    hybrid = HybridDetector(device="cuda:0")
    embedder = SigLIPEmbedder(device="cuda:0")

    total_dropped = Counter()
    all_merges = []

    for clip in DEV_CLIPS:
        cap = cv2.VideoCapture(clip)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        cap.release()

        tracks, _, dropped, merges = process_video_tracks_phase1b(
            video_path=clip,
            detector=hybrid,
            start_time_iso="2026-10-08T18:00:00",
            fps=fps,
            duration_s=dur,
            stride=2,
            tracker_name="bytetrack_tuned",
            embedder=embedder
        )

        for reason, count in dropped.items():
            total_dropped[reason] += count

        for m in merges:
            m["video"] = clip
            all_merges.append(m)

    print("\n--- Track Filter Dropped Tracks Breakdown ---")
    print(f"  Dropped due to short duration (< 1.0s) : {total_dropped['duration']:4d}")
    print(f"  Dropped due to low hits (< 4 observations): {total_dropped['hits']:4d}")
    print(f"  Dropped due to low mean confidence (<0.35): {total_dropped['mean_conf']:4d}")
    print(f"  Total Raw Fragment Tracks Filtered Out   : {total_dropped['total_dropped']:4d}")

    print(f"\n--- Stitched Track Pairs Found: {len(all_merges)} total ---")
    print("Top 10 Stitched Pairs for Visual Verification:")
    print(f"{'#':2s} | {'Video Clip':28s} | {'Trk A':5s} | {'Trk B':5s} | {'Label':8s} | {'Group':7s} | {'Gap (s)':7s} | {'Center Dist':11s} | {'Cosine Sim':10s}")
    print("-" * 105)

    sample_merges = all_merges[:10]
    for idx, m in enumerate(sample_merges, 1):
        print(f"{idx:2d} | {m['video']:28s} | {m['base_id']:5d} | {m['cand_id']:5d} | {m['label']:8s} | {m['group']:7s} | {m['gap_s']:7.3f} | {m['center_dist']:11.4f} | {m['cosine_sim']:10.4f}")

    return total_dropped, all_merges


def run_experiment_5_color():
    """
    5. Color: Lab k-means vs SigLIP color prompts on 30 sample crops.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: COLOR RECOGNITION EVALUATION (Lab K-Means vs SigLIP Prompts)")
    print("=" * 80)

    # Collect 30 distinct object crops from the dev clips
    hybrid = HybridDetector(device="cuda:0")
    crops: List[Tuple[np.ndarray, str, str, str]] = [] # (crop, true_label, clip, frame)

    for clip in DEV_CLIPS:
        if len(crops) >= 30:
            break
        cap = cv2.VideoCapture(clip)
        f_idx = 0
        while cap.isOpened() and len(crops) < 30:
            ret, frame = cap.read()
            if not ret:
                break
            if f_idx % 15 == 0:
                dets = hybrid.detect(frame, imgsz=640)
                for d in dets:
                    bx1, by1, bx2, by2 = d["bbox"]
                    if (bx2 - bx1) >= 25 and (by2 - by1) >= 25:
                        c_img = frame[by1:by2, bx1:bx2].copy()
                        crops.append((c_img, d["label"], clip, f"f_{f_idx}"))
                        if len(crops) >= 30:
                            break
            f_idx += 1
        cap.release()

    crops = crops[:30]
    print(f"Extracted {len(crops)} crops from dev clips.")

    # Candidate colors
    COLOR_CANDIDATES = ["red", "blue", "green", "yellow", "black", "white", "silver", "grey", "orange", "brown"]

    # SigLIP color prompt zero-shot classifier
    embedder = SigLIPEmbedder(device="cuda:0")
    color_prompts = [f"a photo of a {c} object" for c in COLOR_CANDIDATES]
    color_text_embs = embedder.embed_text(color_prompts)

    padded_crops = [pad_to_square(c[0]) for c in crops]
    crop_embs = embedder.embed_images(padded_crops, initial_batch_size=16)

    # Predictions
    lab_predictions = []
    siglip_predictions = []

    print("\nEvaluating 30 Crops:")
    print(f"{'#':2s} | {'Object Label':12s} | {'Source Clip':26s} | {'Lab K-Means (Top 2)':30s} | {'SigLIP Top Color':16s} | {'Agreement':10s}")
    print("-" * 110)

    agreement_count = 0
    raw_color_results = []

    for i in range(len(crops)):
        crop_img, lbl, clip_src, f_src = crops[i]
        group = get_object_group(lbl)
        
        # 1. Lab K-Means prediction
        cdict = extract_track_colors(crop_img, lbl, group)
        # Flatten top colors
        top_lab_colors = []
        for part, clist in cdict.items():
            for item in clist:
                top_lab_colors.append((item["color"], item["fraction"]))
        top_lab_colors.sort(key=lambda x: x[1], reverse=True)
        primary_lab = top_lab_colors[0][0] if top_lab_colors else "unknown"
        lab_str = ", ".join([f"{c}:{f:.2f}" for c, f in top_lab_colors[:2]])

        # 2. SigLIP zero-shot text prediction
        sims = np.dot(color_text_embs, crop_embs[i])
        best_idx = int(np.argmax(sims))
        siglip_color = COLOR_CANDIDATES[best_idx]

        # Agreement: if primary lab matches siglip OR siglip is in top 2 lab colors
        lab_color_names = [c[0] for c in top_lab_colors[:2]]
        is_agree = (primary_lab == siglip_color) or (siglip_color in lab_color_names)
        if is_agree:
            agreement_count += 1

        print(f"{i+1:2d} | {lbl:12s} | {clip_src:26s} | {lab_str:30s} | {siglip_color:16s} | {'AGREE' if is_agree else 'DIFF':10s}")
        raw_color_results.append({
            "idx": i + 1,
            "label": lbl,
            "clip": clip_src,
            "lab": lab_str,
            "siglip": siglip_color,
            "agree": is_agree
        })

    agreement_pct = (agreement_count / len(crops)) * 100.0
    print(f"\nColor Method Agreement: {agreement_count}/{len(crops)} ({agreement_pct:.1f}%)")
    return raw_color_results, agreement_pct


def run_experiment_6_reingest():
    """
    6. Re-ingest into new index folder: index_phase1b.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: RE-INGESTION INTO index_phase1b")
    print("=" * 80)

    index_dir = Path("index_phase1b")
    db_path = index_dir / "index.db"
    snapshots_dir = index_dir / "snapshots"

    pipeline = IngestPipeline(
        db_path=db_path,
        snapshots_dir=snapshots_dir,
        use_hybrid_detector=True
    )

    ingest_results = []
    for clip in DEV_CLIPS:
        print(f"\nIndexing: {clip} into {index_dir}...")
        res = pipeline.ingest_video(
            video_path=clip,
            stride=2,
            tracker_name="bytetrack_tuned",
            force=True
        )
        ingest_results.append(res)
        print(f"  Indexed: {res['camera']} | Tracks: {res['num_tracks']} | Time: {res['wall_time_s']}s")

    conn = get_db(db_path)
    cur = conn.execute("SELECT count(*) FROM tracks;")
    n_tracks = cur.fetchone()[0]
    cur2 = conn.execute("SELECT count(*) FROM frames;")
    n_frames = cur2.fetchone()[0]

    # Inspect sample Phase 1b track columns
    cur3 = conn.execute("SELECT id, label, \"group\", quality, hits, colors FROM tracks LIMIT 5;")
    sample_rows = [dict(r) for r in cur3.fetchall()]

    print(f"\nFinal index_phase1b DB Summary:")
    print(f"  Total Tracks: {n_tracks}")
    print(f"  Total Frames: {n_frames}")
    print("\nSample Track Rows with Phase 1b Schema Columns:")
    for r in sample_rows:
        print(" ", r)

    return ingest_results, n_tracks, n_frames


def main():
    print("Starting MULTIStream Phase 1b Accuracy Fixes Evaluation Suite...")
    t_start = time.perf_counter()

    det_a, det_b = run_experiment_1_detector()
    strides, bot_fps, tracker_comp = run_experiment_2_tracker_speed_and_tracks()
    dropped, merges = run_experiment_3_and_4_track_filter_and_stitching()
    colors, agree_pct = run_experiment_5_color()
    ingest_res, n_trk, n_frm = run_experiment_6_reingest()

    total_time = time.perf_counter() - t_start
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS COMPLETED IN {total_time:.2f}s")
    print("=" * 80)


if __name__ == "__main__":
    main()
