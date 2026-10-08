import os
import cv2
import json
import time
import uuid
import datetime
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np

from src.index.db import get_db, insert_video, insert_track, emb_to_blob
from src.utils.color import extract_dominant_color
from src.ingest.detect_track import reset_tracker, create_annotated_snapshot
from src.utils.vram import get_nvml_vram_info

logger = logging.getLogger("multistream.live")


class LiveStreamWorker:
    """
    Real-Time IP Webcam / Mobile Stream Ingest Worker.
    Connects to live RTSP/HTTP phone streams (e.g., IP Webcam on Android/iOS or simulated streams),
    performs real-time detection, tracking, dominant color classification, SigLIP embedding,
    and updates SQLite index on-the-fly.
    """
    def __init__(
        self,
        stream_url: str,
        camera_name: str = "mobile_cam01",
        db_path: str | Path = "index_base/index.db",
        snapshots_dir: str | Path = "index_base/snapshots",
        recorded_dir: str | Path = "footage/recorded",
        stride: int = 5,
        conf_thresh: float = 0.25,
        imgsz: int = 640
    ):
        self.stream_url = stream_url
        self.camera_name = camera_name
        self.db_path = Path(db_path)
        self.snapshots_dir = Path(snapshots_dir) / camera_name
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.recorded_dir = Path(recorded_dir)
        self.recorded_dir.mkdir(parents=True, exist_ok=True)

        self.stride = stride
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        self.is_running = False
        self.thread: Optional[threading.Thread] = None

        # Stats
        self.stats = {
            "camera": self.camera_name,
            "stream_url": self.stream_url,
            "status": "idle",
            "frames_read": 0,
            "tracks_indexed": 0,
            "active_tracks": 0,
            "fps": 0.0,
            "start_time": None,
            "latest_track": None,
            "latest_snapshot": None
        }

    def start(self, yolo_model, embedder, classes: List[str]):
        """Start live stream ingest thread."""
        if self.is_running:
            return
        self.is_running = True
        self.yolo = yolo_model
        self.embedder = embedder
        self.classes = classes

        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"[LIVE] Started live stream worker for camera '{self.camera_name}' on '{self.stream_url}'")

    def stop(self):
        """Signal worker thread to stop."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.stats["status"] = "stopped"
        logger.info(f"[LIVE] Stopped live stream worker for camera '{self.camera_name}'")

    def _run_loop(self):
        self.stats["status"] = "connecting"
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            logger.error(f"[LIVE] Could not open stream: {self.stream_url}")
            self.stats["status"] = "error: could not connect to stream"
            self.is_running = False
            return

        self.stats["status"] = "streaming"
        start_dt = datetime.datetime.now()
        start_iso = start_dt.isoformat()
        self.stats["start_time"] = start_iso

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

        # Video recording file for on-demand clip extraction
        rec_filename = f"{self.camera_name}_{start_dt.strftime('%Y%m%d_%H%M%S')}.mp4"
        rec_path = self.recorded_dir / rec_filename
        rel_rec_path = f"footage/recorded/{rec_filename}".replace("\\", "/")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(rec_path), fourcc, fps, (width, height))

        # Register video in DB
        conn = get_db(self.db_path)
        insert_video(
            conn=conn,
            path=rel_rec_path,
            camera=self.camera_name,
            start_time=start_iso,
            fps=fps,
            width=width,
            height=height,
            duration_s=0.0,
            start_source="live_stream",
            indexed_at=start_iso,
            rotation=0
        )

        reset_tracker(self.yolo)
        frame_count = 0
        t0 = time.time()
        tracks_seen = set()

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    # In file simulation loop back to start; in live stream wait or break
                    if not self.stream_url.lower().startswith(("http://", "rtsp://")):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        time.sleep(0.05)
                        continue

                frame_count += 1
                self.stats["frames_read"] = frame_count
                writer.write(frame)

                current_time = time.time()
                elapsed = current_time - t0
                if elapsed > 0:
                    self.stats["fps"] = round(frame_count / elapsed, 1)

                offset_s = round(frame_count / fps, 3)
                curr_dt = start_dt + datetime.timedelta(seconds=offset_s)
                curr_iso = curr_dt.isoformat()

                if frame_count % self.stride == 0:
                    # Run YOLO tracking
                    results = self.yolo.track(
                        frame,
                        persist=True,
                        conf=self.conf_thresh,
                        imgsz=self.imgsz,
                        classes=list(range(len(self.classes))),
                        verbose=False
                    )

                    if results and len(results) > 0 and results[0].boxes is not None and results[0].boxes.id is not None:
                        boxes = results[0].boxes
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        confs = boxes.conf.cpu().numpy().astype(float)
                        xyxy = boxes.xyxy.cpu().numpy().astype(int)
                        track_ids = boxes.id.cpu().numpy().astype(int)

                        for cls_id, conf, box_px, trk_id in zip(cls_ids, confs, xyxy, track_ids):
                            label = self.classes[cls_id] if cls_id < len(self.classes) else "object"
                            global_trk_id = f"{self.camera_name}_trk_{trk_id}"

                            bx1, by1, bx2, by2 = int(box_px[0]), int(box_px[1]), int(box_px[2]), int(box_px[3])
                            bx1, by1 = max(0, bx1), max(0, by1)
                            bx2, by2 = min(width, bx2), min(height, by2)
                            crop = frame[by1:by2, bx1:bx2]

                            if crop.size == 0:
                                continue

                            # Detect dominant color
                            detected_color, color_conf = extract_dominant_color(crop)

                            # Save annotated snapshot
                            snap_filename = f"track_{trk_id}.jpg"
                            snap_path = self.snapshots_dir / snap_filename
                            rel_snap_path = f"snapshots/{self.camera_name}/{snap_filename}".replace("\\", "/")

                            annotated = create_annotated_snapshot(
                                frame,
                                [bx1, by1, bx2, by2],
                                f"{detected_color} {label}",
                                trk_id,
                                conf
                            )
                            cv2.imwrite(str(snap_path), annotated)

                            # Compute crop embedding
                            crop_emb = self.embedder.embed_images([crop])[0]
                            emb_blob = emb_to_blob(crop_emb)

                            # Normalized bbox
                            bbox_norm = [
                                round(bx1 / width, 4),
                                round(by1 / height, 4),
                                round(bx2 / width, 4),
                                round(by2 / height, 4)
                            ]

                            # Insert / Update track in SQLite
                            track_rec = {
                                "id": global_trk_id,
                                "video": rel_rec_path,
                                "camera": self.camera_name,
                                "track_id": int(trk_id),
                                "label": label,
                                "t_start": curr_iso,
                                "t_end": curr_iso,
                                "t_best": curr_iso,
                                "offset_start": offset_s,
                                "offset_end": offset_s,
                                "offset_best": offset_s,
                                "bbox_px": json.dumps([bx1, by1, bx2, by2]),
                                "bbox_norm": json.dumps(bbox_norm),
                                "snapshot": rel_snap_path,
                                "emb": emb_blob
                            }
                            insert_track(conn, track_rec)

                            if trk_id not in tracks_seen:
                                tracks_seen.add(trk_id)
                                self.stats["tracks_indexed"] = len(tracks_seen)
                                self.stats["latest_track"] = {
                                    "id": global_trk_id,
                                    "label": label,
                                    "color": detected_color,
                                    "timestamp": curr_iso,
                                    "offset_s": offset_s
                                }
                                self.stats["latest_snapshot"] = rel_snap_path

        except Exception as e:
            logger.error(f"[LIVE] Error during stream processing: {e}")
            self.stats["status"] = f"error: {e}"
        finally:
            cap.release()
            writer.release()
            # Update video duration in DB
            with conn:
                conn.execute(
                    "UPDATE videos SET duration_s = ? WHERE path = ?;",
                    (round(frame_count / fps, 2), rel_rec_path)
                )
            self.stats["status"] = "finished"
            logger.info(f"[LIVE] Finished stream for '{self.camera_name}'. Total tracks: {len(tracks_seen)}")
