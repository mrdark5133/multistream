import os
import cv2
import json
import yaml
import time
import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from ultralytics import YOLO

from src.index.db import (
    init_db,
    get_db,
    set_config,
    get_video,
    insert_video,
    insert_tracks_batch,
    insert_frames_batch,
    emb_to_blob
)
from src.ingest.start_time import resolve_video_metadata, VideoMetadata
from src.ingest.detect_track import (
    load_yolo_world,
    process_video_tracks,
    TrackSummary,
    SampledFrame
)
from src.ingest.embed import SigLIPEmbedder
from src.utils.vram import get_nvml_vram_info


class IngestPipeline:
    """
    Multi-camera video ingest orchestrator with incremental indexing and duplicate prevention.
    """
    def __init__(
        self,
        db_path: str | Path,
        snapshots_dir: str | Path,
        embedder_name: str = "google/siglip-base-patch16-224",
        yolo_weights: str = "yolov8s-worldv2.pt",
        vocab_path: str = "config/vocab.yaml",
        device: str = "cuda:0",
        yolo_model: Optional[YOLO] = None,
        embedder: Optional[SigLIPEmbedder] = None
    ):
        self.db_path = Path(db_path)
        self.snapshots_dir = Path(snapshots_dir)
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.embedder_name = embedder_name
        self.device = device
        
        # Initialize SQLite DB
        self.conn = init_db(self.db_path)
        set_config(self.conn, "embedder", embedder_name)
        
        # Load or reuse models
        if yolo_model is not None:
            self.yolo = yolo_model
            with open(vocab_path, "r", encoding="utf-8") as f:
                self.classes = yaml.safe_load(f)["classes"]
        else:
            self.yolo, self.classes = load_yolo_world(
                weights_path=yolo_weights,
                vocab_path=vocab_path,
                device=device
            )
            
        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = SigLIPEmbedder(
                model_id=embedder_name,
                device=device
            )

    def ingest_video(
        self,
        video_path: str | Path,
        manifest_path: Optional[str | Path] = None,
        manual_camera: Optional[str] = None,
        manual_start: Optional[str] = None,
        stride: int = 5,
        conf_thresh: float = 0.25,
        imgsz: int = 640,
        frame_sample_interval_s: float = 2.0,
        force: bool = False
    ) -> Dict[str, Any]:
        """
        Ingest single video file into the index.
        Duplicate prevention: skips video if already indexed unless force=True.
        """
        v_path = Path(video_path)
        v_path_str = str(v_path).replace("\\", "/")
        stem = v_path.stem
        
        t0 = time.perf_counter()
        vram_total, vram_used_start, vram_free_start = get_nvml_vram_info()
        
        # 1. Duplicate check (Task 1.7)
        existing = get_video(self.conn, v_path_str)
        if existing and not force:
            print(f"[Duplicate Prevention] Skipping {v_path_str}: already indexed at {existing['indexed_at']}")
            return {
                "status": "skipped",
                "video": v_path_str,
                "reason": "already_indexed",
                "indexed_at": existing["indexed_at"]
            }

        # 2. Metadata resolution (Task 1.2 & 1.10)
        meta: VideoMetadata = resolve_video_metadata(
            video_path=v_path,
            manifest_path=manifest_path,
            manual_start_time=manual_start,
            manual_camera=manual_camera
        )

        # 3. Clean up any previous records for this video if forced re-index
        if force:
            with self.conn:
                self.conn.execute("DELETE FROM tracks WHERE video = ?;", (v_path_str,))
                self.conn.execute("DELETE FROM frames WHERE video = ?;", (v_path_str,))

        # 4. Detection & Tracking (Tasks 1.3, 1.4, 1.9, 1.10)
        tracks, sampled_frames = process_video_tracks(
            video_path=v_path,
            yolo_model=self.yolo,
            classes=self.classes,
            start_time_iso=meta.start_time,
            fps=meta.fps,
            duration_s=meta.duration_s,
            stride=stride,
            conf_thresh=conf_thresh,
            imgsz=imgsz,
            rotation=meta.rotation,
            frame_sample_interval_s=frame_sample_interval_s
        )

        # 5. Crop embeddings with OOM backoff (Task 1.5)
        crop_images = [t.best_crop for t in tracks]
        crop_embs = self.embedder.embed_images(crop_images, initial_batch_size=16)

        # 6. Whole-frame embeddings (Task 1.6)
        frame_images = [f.frame_bgr for f in sampled_frames]
        frame_embs = self.embedder.embed_images(frame_images, initial_batch_size=8)

        # 7. Save snapshots and build DB records
        video_snapshots_dir = self.snapshots_dir / stem
        video_snapshots_dir.mkdir(parents=True, exist_ok=True)
        
        track_records = []
        for i, t in enumerate(tracks):
            snap_rel_path = f"snapshots/{stem}/track_{t.track_id}.jpg"
            snap_abs_path = self.snapshots_dir.parent / snap_rel_path
            cv2.imwrite(str(snap_abs_path), t.best_snapshot_bgr)
            
            emb_blob = emb_to_blob(crop_embs[i])
            track_records.append({
                "id": f"{stem}_trk_{t.track_id}",
                "video": v_path_str,
                "camera": meta.camera,
                "track_id": t.track_id,
                "label": t.label,
                "t_start": t.t_start,
                "t_end": t.t_end,
                "t_best": t.t_best,
                "offset_start": t.offset_start,
                "offset_end": t.offset_end,
                "bbox_px": json.dumps(t.bbox_px),
                "bbox_norm": json.dumps(t.bbox_norm),
                "snapshot": snap_rel_path.replace("\\", "/"),
                "emb": emb_blob
            })

        frame_records = []
        for i, f in enumerate(sampled_frames):
            snap_rel_path = f"snapshots/{stem}/frame_{f.frame_idx}.jpg"
            snap_abs_path = self.snapshots_dir.parent / snap_rel_path
            
            # Save frame snapshot resized to max 640px
            fh, fw = f.frame_bgr.shape[:2]
            scale = min(640 / max(fw, fh), 1.0)
            if scale < 1.0:
                small_frame = cv2.resize(f.frame_bgr, (int(fw * scale), int(fh * scale)), interpolation=cv2.INTER_AREA)
            else:
                small_frame = f.frame_bgr
            cv2.imwrite(str(snap_abs_path), small_frame)
            
            emb_blob = emb_to_blob(frame_embs[i])
            frame_records.append({
                "id": f"{stem}_frm_{f.frame_idx}",
                "video": v_path_str,
                "camera": meta.camera,
                "t_abs": f.iso_time,
                "offset_s": f.offset_s,
                "snapshot": snap_rel_path.replace("\\", "/"),
                "emb": emb_blob
            })

        # 8. Insert into SQLite DB (Task 1.1)
        now_iso = datetime.datetime.now().isoformat()
        insert_video(
            self.conn,
            path=v_path_str,
            camera=meta.camera,
            start_time=meta.start_time,
            fps=meta.fps,
            width=meta.width,
            height=meta.height,
            duration_s=meta.duration_s,
            start_source=meta.start_source,
            indexed_at=now_iso,
            rotation=meta.rotation
        )

        insert_tracks_batch(self.conn, track_records)
        insert_frames_batch(self.conn, frame_records)

        elapsed_s = time.perf_counter() - t0
        _, vram_used_end, _ = get_nvml_vram_info()
        
        return {
            "status": "indexed",
            "video": v_path_str,
            "camera": meta.camera,
            "start_time": meta.start_time,
            "start_source": meta.start_source,
            "rotation": meta.rotation,
            "duration_s": meta.duration_s,
            "num_tracks": len(tracks),
            "num_frames": len(sampled_frames),
            "wall_time_s": round(elapsed_s, 2),
            "vram_used_mib": vram_used_end
        }
