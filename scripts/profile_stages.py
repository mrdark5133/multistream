import time
import json
import sys
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.detector import HybridDetector, get_object_group
from src.ingest.detect_track import (
    rotate_frame,
    SampledFrame,
    create_annotated_snapshot
)
from src.ingest.track_engine import (
    create_tracker,
    run_tracker_on_detections,
    compute_crop_quality,
    filter_tracks,
    stitch_tracks,
    pad_to_square,
    TrackState,
    TrackObservation as EngineTrackObs
)
from src.utils.lab_color import extract_track_colors
from src.ingest.embed import SigLIPEmbedder
from src.index.db import (
    init_db,
    insert_video,
    insert_tracks_batch,
    insert_frames_batch,
    emb_to_blob
)


def profile_single_run(video_path: str, hybrid: HybridDetector, embedder: SigLIPEmbedder) -> Dict[str, float]:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stride = 2
    rotation = 0

    t_decode_rotate = 0.0
    t_detect = 0.0
    t_track = 0.0
    t_color = 0.0
    t_stitching = 0.0
    t_snapshots = 0.0
    t_crop_emb = 0.0
    t_frame_emb = 0.0
    t_db_write = 0.0

    tracker = create_tracker("bytetrack_tuned")
    raw_tracks: Dict[int, TrackState] = {}
    sampled_frames: List[SampledFrame] = []
    last_frame_sample_time = -999.0
    frame_idx = 0

    while True:
        t0 = time.perf_counter()
        ret, raw_frame = cap.read()
        if not ret:
            break
        if rotation != 0:
            frame = rotate_frame(raw_frame, rotation)
        else:
            frame = raw_frame
        offset_s = frame_idx / fps
        t_decode_rotate += (time.perf_counter() - t0)

        # Whole frame sampling every 2.0s
        if (offset_s - last_frame_sample_time) >= 2.0:
            sf_h, sf_w = frame.shape[:2]
            scale = min(640 / max(sf_w, sf_h), 1.0)
            small_f = cv2.resize(frame, (int(sf_w * scale), int(sf_h * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else frame.copy()
            sampled_frames.append(SampledFrame(
                frame_idx=frame_idx,
                offset_s=round(offset_s, 3),
                iso_time=f"2026-10-08T18:00:{offset_s:06.3f}",
                frame_bgr=small_f
            ))
            last_frame_sample_time = offset_s

        if frame_idx % stride == 0:
            # Detection
            t_det0 = time.perf_counter()
            dets = hybrid.detect(frame, imgsz=640)
            t_detect += (time.perf_counter() - t_det0)

            # Tracking
            t_trk0 = time.perf_counter()
            tracked_dets = run_tracker_on_detections(tracker, dets, frame, w, h)
            t_track += (time.perf_counter() - t_trk0)

            for td in tracked_dets:
                tid = td["track_id"]
                bx1, by1, bx2, by2 = td["bbox"]
                conf = td["conf"]
                label = td["label"]
                if (bx2 - bx1) <= 0 or (by2 - by1) <= 0:
                    continue
                crop = frame[by1:by2, bx1:bx2].copy()
                quality = compute_crop_quality(crop, conf, [bx1, by1, bx2, by2], w, h)
                bbox_norm = [round(bx1 / w, 4), round(by1 / h, 4), round(bx2 / w, 4), round(by2 / h, 4)]
                obs = EngineTrackObs(
                    frame_idx=frame_idx,
                    offset_s=round(offset_s, 3),
                    iso_time=f"2026-10-08T18:00:{offset_s:06.3f}",
                    bbox_px=[bx1, by1, bx2, by2],
                    bbox_norm=bbox_norm,
                    conf=conf,
                    quality=quality,
                    crop_bgr=crop
                )
                if tid not in raw_tracks:
                    raw_tracks[tid] = TrackState(track_id=tid)
                raw_tracks[tid].add(obs, label, conf, frame_bgr=frame)

        frame_idx += 1
    cap.release()

    # Filtering
    filtered_states, _ = filter_tracks(raw_tracks, min_duration_s=1.0, min_hits=4, min_mean_conf=0.35)

    # Color extraction
    t_col0 = time.perf_counter()
    track_crops = []
    track_padded_crops = []
    for st in filtered_states:
        best_obs, _ = st.get_best_observation()
        raw_crop = st.best_crop_bgr if st.best_crop_bgr is not None else best_obs.crop_bgr
        padded = pad_to_square(raw_crop)
        track_padded_crops.append(padded)
        voted_label = st.get_voted_label()
        extract_track_colors(padded, voted_label, get_object_group(voted_label))
    t_color += (time.perf_counter() - t_col0)

    # Crop embedding for stitching
    t_c_emb0 = time.perf_counter()
    embs = embedder.embed_images(track_padded_crops, initial_batch_size=16)
    t_crop_emb += (time.perf_counter() - t_c_emb0)

    # Stitching
    t_st0 = time.perf_counter()
    stitched_states, stitched_embs, merge_logs = stitch_tracks(
        filtered_states,
        embs,
        max_gap_s=2.0,
        max_center_dist=0.25,
        min_cosine_sim=0.80
    )
    t_stitching += (time.perf_counter() - t_st0)

    # Frame embedding
    t_f_emb0 = time.perf_counter()
    frame_images = [f.frame_bgr for f in sampled_frames]
    frame_embs = embedder.embed_images(frame_images, initial_batch_size=8)
    t_frame_emb += (time.perf_counter() - t_f_emb0)

    # Snapshots saving
    t_snap0 = time.perf_counter()
    temp_dir = Path("eval/profile_temp_snaps")
    temp_dir.mkdir(parents=True, exist_ok=True)
    for st in stitched_states:
        best_obs, _ = st.get_best_observation()
        voted_label = st.get_voted_label()
        snap = create_annotated_snapshot(st.best_snapshot_bgr, best_obs.bbox_px, voted_label, st.track_id, best_obs.conf, max_dim=640)
        cv2.imwrite(str(temp_dir / f"snap_{st.track_id}.jpg"), snap)
    t_snapshots += (time.perf_counter() - t_snap0)

    # DB write
    t_db0 = time.perf_counter()
    db_file = Path("eval/profile_temp.db")
    if db_file.exists():
        db_file.unlink()
    conn = init_db(db_file)
    insert_video(
        conn,
        path=video_path,
        camera="cam_landscape",
        start_time="2026-10-08T18:00:00",
        fps=fps,
        width=w,
        height=h,
        duration_s=len(sampled_frames) * 2.0,
        start_source="filename",
        indexed_at="2026-10-08T18:00:00",
        rotation=0
    )
    track_records = []
    for i, st in enumerate(stitched_states):
        best_obs, _ = st.get_best_observation()
        track_records.append({
            "id": f"prof_trk_{st.track_id}",
            "video": video_path,
            "camera": "cam_landscape",
            "track_id": st.track_id,
            "label": st.get_voted_label(),
            "group": get_object_group(st.get_voted_label()),
            "colors": json.dumps({}),
            "quality": 0.5,
            "hits": st.hits,
            "t_start": st.observations[0].iso_time,
            "t_end": st.observations[-1].iso_time,
            "t_best": best_obs.iso_time,
            "offset_start": st.observations[0].offset_s,
            "offset_end": st.observations[-1].offset_s,
            "offset_best": best_obs.offset_s,
            "bbox_px": json.dumps(best_obs.bbox_px),
            "bbox_norm": json.dumps(best_obs.bbox_norm),
            "snapshot": f"snapshots/prof_{st.track_id}.jpg",
            "emb": emb_to_blob(stitched_embs[i])
        })
    insert_tracks_batch(conn, track_records)
    frame_records = []
    for i, sf in enumerate(sampled_frames):
        frame_records.append({
            "id": f"prof_frm_{sf.frame_idx}",
            "video": video_path,
            "camera": "cam_landscape",
            "t_abs": sf.iso_time,
            "offset_s": sf.offset_s,
            "snapshot": f"snapshots/prof_frame_{sf.frame_idx}.jpg",
            "emb": emb_to_blob(frame_embs[i])
        })
    insert_frames_batch(conn, frame_records)
    conn.close()
    t_db_write += (time.perf_counter() - t_db0)

    total_time = (t_decode_rotate + t_detect + t_track + t_snapshots + 
                  t_crop_emb + t_frame_emb + t_db_write + t_color + t_stitching)

    return {
        "decode_and_rotation": t_decode_rotate,
        "detection": t_detect,
        "tracking": t_track,
        "snapshots": t_snapshots,
        "crop_embedding": t_crop_emb,
        "frame_embedding": t_frame_emb,
        "db_write": t_db_write,
        "color": t_color,
        "stitching": t_stitching,
        "total": total_time
    }


def main():
    print("=" * 80)
    print("PROFILING test_landscape.mp4 (3 RUNS, MEDIAN PER STAGE)")
    print("=" * 80)

    hybrid = HybridDetector(device="cuda:0")
    embedder = SigLIPEmbedder(device="cuda:0")
    video_path = "footage/test_landscape.mp4"

    runs = []
    for r in range(3):
        print(f"\n--- Run {r+1}/3 ---")
        t0 = time.perf_counter()
        res = profile_single_run(video_path, hybrid, embedder)
        runs.append(res)
        print(f"Run {r+1} completed in {time.perf_counter() - t0:.2f}s")
        for k, v in res.items():
            print(f"  {k:22s}: {v:6.3f}s")

    stages = [
        "decode_and_rotation",
        "detection",
        "tracking",
        "snapshots",
        "crop_embedding",
        "frame_embedding",
        "db_write",
        "color",
        "stitching",
        "total"
    ]

    print("\n" + "=" * 80)
    print(f"{'Pipeline Stage':24s} | {'Run 1 (s)':9s} | {'Run 2 (s)':9s} | {'Run 3 (s)':9s} | {'Median (s)':10s} | {'Pct Total':9s}")
    print("-" * 80)

    med_total = float(np.median([r["total"] for r in runs]))

    for s in stages:
        vals = [r[s] for r in runs]
        med = float(np.median(vals))
        pct = (med / med_total) * 100 if s != "total" else 100.0
        print(f"{s:24s} | {vals[0]:9.3f} | {vals[1]:9.3f} | {vals[2]:9.3f} | {med:10.3f} | {pct:8.1f}%")

    # Clean up temp
    import shutil
    shutil.rmtree("eval/profile_temp_snaps", ignore_errors=True)
    Path("eval/profile_temp.db").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
