import cv2
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector, get_object_group
from src.ingest.detect_track import process_video_tracks_phase1b
from src.ingest.embed import SigLIPEmbedder
from src.index.db import get_db

DEV_CLIPS = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4",
    "footage/test_landscape.mp4",
    "footage/test_landscape2.mp4"
]

HAND_COUNTED_REAL_OBJECTS = {
    "footage/test_video01.mp4": 1,   # 1 primary moving pedestrian
    "footage/test_video02.mp4": 1,   # 1 moving animal
    "footage/test_video03.mp4": 3,   # 1 motorcycle + 2 riders
    "footage/test_landscape.mp4": 8, # 8 moving road vehicles
    "footage/test_landscape2.mp4": 6 # 6 moving road vehicles
}

def main():
    hybrid = HybridDetector(device="cuda:0")
    embedder = SigLIPEmbedder(device="cuda:0")

    all_merges = []
    clip_stats = {}

    print("=" * 80)
    print("RERUNNING STITCHING WITH 1-TO-1 ENFORCEMENT & 0.88 VEHICLE THRESHOLD")
    print("=" * 80)

    for clip in DEV_CLIPS:
        stem = Path(clip).stem
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

        n_after_stitch = len(tracks)
        n_after_filter = n_after_stitch + len(merges)
        real_n = HAND_COUNTED_REAL_OBJECTS[clip]

        clip_stats[clip] = {
            "real_n": real_n,
            "after_filter": n_after_filter,
            "after_stitch": n_after_stitch,
            "merges": len(merges),
            "ratio_filtered": round(n_after_filter / real_n, 2),
            "ratio_stitched": round(n_after_stitch / real_n, 2),
            "dropped": dropped
        }

        for m in merges:
            m["clip"] = clip
            m["base_snapshot"] = f"snapshots/{stem}/track_{m['base_id']}.jpg"
            m["cand_snapshot"] = f"snapshots/{stem}/track_{m['cand_id']}.jpg"
            all_merges.append(m)

    print("\n--- Summary of Merged Pairs ---")
    print(f"Total merged pairs found: {len(all_merges)}")
    for idx, m in enumerate(all_merges, 1):
        print(f"Pair {idx}:")
        print(f"  Clip          : {m['clip']}")
        print(f"  Track A (Base): {m['base_id']} -> Snapshot: {m['base_snapshot']}")
        print(f"  Track B (Cand): {m['cand_id']} -> Snapshot: {m['cand_snapshot']}")
        print(f"  Label / Group : {m['label']} ({m['group']})")
        print(f"  Gap (s)       : {m['gap_s']:.3f} s")
        print(f"  Center Dist   : {m['center_dist']:.4f}")
        print(f"  Cosine Sim    : {m['cosine_sim']:.4f}")

    print("\n--- Tracks Per Real Object (Filtered vs Stitched) ---")
    print(f"{'Clip':28s} | {'Real Obj':8s} | {'Filtered':8s} | {'Trk/Obj (Filt)':14s} | {'Stitched':8s} | {'Trk/Obj (Stitch)':16s}")
    print("-" * 95)
    for clip, s in clip_stats.items():
        print(f"{clip:28s} | {s['real_n']:8d} | {s['after_filter']:8d} | {s['ratio_filtered']:14.2f} | {s['after_stitch']:8d} | {s['ratio_stitched']:16.2f}")

    with open("eval/stitching_rerun_results.json", "w") as f:
        json.dump({"merges": all_merges, "stats": clip_stats}, f, indent=2)

if __name__ == "__main__":
    main()
