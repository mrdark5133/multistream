import os
import cv2
import json
import time
import uuid
import datetime
import threading
import logging
import subprocess
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
        conf_thresh: float = 0.15,
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
        self.latest_jpeg: Optional[bytes] = None
        self._fallback_counter = 0
        self._active_fallback_boxes: Dict[int, Any] = {}

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
            "recorded_file": None,
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

    def _assign_fallback_id(self, box_xyxy) -> int:
        """Assign or maintain spatial fallback track ID when ByteTrack yields no ID."""
        best_id = None
        best_iou = 0.25
        for trk_id, prev_box in list(self._active_fallback_boxes.items()):
            ix1 = max(box_xyxy[0], prev_box[0])
            iy1 = max(box_xyxy[1], prev_box[1])
            ix2 = min(box_xyxy[2], prev_box[2])
            iy2 = min(box_xyxy[3], prev_box[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area1 = (box_xyxy[2] - box_xyxy[0]) * (box_xyxy[3] - box_xyxy[1])
            area2 = (prev_box[2] - prev_box[0]) * (prev_box[3] - prev_box[1])
            union = area1 + area2 - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_id = trk_id
        if best_id is None:
            self._fallback_counter += 1
            best_id = self._fallback_counter
        self._active_fallback_boxes[best_id] = box_xyxy
        return best_id

    def _run_loop(self):
        self.stats["status"] = "connecting"
        url = self.stream_url.strip()
        # If user provides IP Webcam root (e.g. http://192.168.X.X:8080 or http://192.168.X.X:8080/),
        # auto-append /video for OpenCV MJPEG stream consumption
        candidate_urls = [url]
        if url.startswith(("http://", "https://")) and not any(url.endswith(s) for s in ["/video", "/mjpg", "/mjpeg", ".mp4", ".mkv", ".ts"]):
            candidate_urls.insert(0, url.rstrip("/") + "/video")

        cap = None
        for cand in candidate_urls:
            logger.info(f"[LIVE] Attempting to open video stream at: {cand}")
            test_cap = cv2.VideoCapture(cand)
            if test_cap.isOpened():
                cap = test_cap
                self.stream_url = cand
                break
            test_cap.release()

        if cap is None or not cap.isOpened():
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
        self.stats["recorded_file"] = rel_rec_path

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
        active_preview_boxes = []

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
                    # Run YOLO tracking with bytetrack
                    results = self.yolo.track(
                        frame,
                        persist=True,
                        tracker="bytetrack.yaml",
                        conf=self.conf_thresh,
                        imgsz=self.imgsz,
                        verbose=False
                    )

                    new_boxes = []
                    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                        boxes = results[0].boxes
                        cls_ids = boxes.cls.cpu().numpy().astype(int)
                        confs = boxes.conf.cpu().numpy().astype(float)
                        xyxy = boxes.xyxy.cpu().numpy().astype(int)

                        if boxes.id is not None:
                            track_ids = boxes.id.cpu().numpy().astype(int)
                        else:
                            track_ids = [self._assign_fallback_id(xyxy[i]) for i in range(len(boxes))]

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
                            new_boxes.append((bx1, by1, bx2, by2, f"#{trk_id} {detected_color} {label}"))

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

                    active_preview_boxes = new_boxes

                # Generate live preview frame for Web UI video feed
                if frame_count % 2 == 0 or self.latest_jpeg is None:
                    disp = frame.copy()
                    for pb in active_preview_boxes:
                        cv2.rectangle(disp, (pb[0], pb[1]), (pb[2], pb[3]), (0, 255, 0), 2)
                        cv2.putText(
                            disp,
                            pb[4],
                            (pb[0], max(18, pb[1] - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                    _, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 55])
                    self.latest_jpeg = buf.tobytes()

        except Exception as e:
            logger.error(f"[LIVE] Error during stream processing: {e}")
            self.stats["status"] = f"error: {e}"
        finally:
            cap.release()
            writer.release()

            # Ensure recorded MP4 is transcoded to standard HTML5/IDE-compatible H.264
            if rec_path.exists() and rec_path.stat().st_size > 0:
                convert_to_h264(rec_path)

            # Update video duration in DB
            with conn:
                conn.execute(
                    "UPDATE videos SET duration_s = ? WHERE path = ?;",
                    (round(frame_count / fps, 2), rel_rec_path)
                )
            self.stats["status"] = "finished"
            logger.info(f"[LIVE] Finished stream for '{self.camera_name}'. Total tracks: {len(tracks_seen)}")


def convert_to_h264(video_path: Path) -> bool:
    """
    Transcode an MPEG-4 (mp4v) recording into web/IDE-standard H.264 (avc1/yuv420p)
    so that VSCode, Chromium, and HTML5 video tags can play it directly.
    """
    winget_ffmpeg = Path(r"C:\Users\Harivarman R\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe")
    ffmpeg_exe = str(winget_ffmpeg) if winget_ffmpeg.exists() else os.environ.get("FFMPEG_BIN", "ffmpeg")

    tmp_path = video_path.with_name(f"temp_{video_path.name}")
    cmd = [
        ffmpeg_exe, "-y",
        "-i", str(video_path),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "ultrafast",
        "-crf", "23",
        "-an",
        str(tmp_path)
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if res.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            tmp_path.replace(video_path)
            logger.info(f"[LIVE] Successfully transcoded {video_path.name} to standard H.264")
            return True
        else:
            logger.warning(f"[LIVE] Transcode returned code {res.returncode}: {res.stderr[:200]}")
    except Exception as e:
        logger.warning(f"[LIVE] Error transcoding {video_path.name}: {e}")
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass
    return False
