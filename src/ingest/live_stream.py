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
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from src.index.db import get_db, insert_video, insert_track, emb_to_blob
from src.utils.color import extract_dominant_color, extract_live_chromatic_color
from src.ingest.detect_track import reset_tracker, create_annotated_snapshot
from src.ingest.track_engine import create_tracker, run_tracker_on_detections
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
        imgsz: int = 640,
        store_live_tracks: bool = False
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
        self.store_live_tracks = store_live_tracks

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

    def extend_vocabulary(self, phrase: str) -> float:
        """
        Dynamically extend the live YOLO-World vocabulary with a new query object phrase.
        Safely re-indexes text prompt features on the target device.
        Returns the delay in milliseconds.
        """
        phrase_clean = phrase.strip().lower()
        if not phrase_clean or phrase_clean in self.classes:
            return 0.0

        t0 = time.perf_counter()
        self.classes.append(phrase_clean)
        if hasattr(self, "yolo") and self.yolo is not None:
            if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "clip_model"):
                self.yolo.model.clip_model = None
            self.yolo.set_classes(self.classes)
        delay_ms = (time.perf_counter() - t0) * 1000.0
        logger.info(f"[LIVE] Extended vocabulary with '{phrase_clean}' in {delay_ms:.2f} ms. Total classes: {len(self.classes)}")
        return delay_ms

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

        if hasattr(self.yolo, "float"):
            self.yolo.float()

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
                    try:
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
                                raw_label = self.classes[cls_id] if cls_id < len(self.classes) else "object"
                                # Requirement 3: store detections above 0.15, but only show detector label above 0.35; otherwise show "object"
                                display_label = raw_label if conf >= 0.35 else "object"
                                global_trk_id = f"{self.camera_name}_trk_{trk_id}"

                                bx1, by1, bx2, by2 = int(box_px[0]), int(box_px[1]), int(box_px[2]), int(box_px[3])
                                bx1, by1 = max(0, bx1), max(0, by1)
                                bx2, by2 = min(width, bx2), min(height, by2)
                                crop = frame[by1:by2, bx1:bx2]

                                if crop.size == 0:
                                    continue

                                # Requirement 4: show a color only if its fraction is above 0.5 and the crop is chromatic; otherwise show none
                                detected_color, color_frac = extract_live_chromatic_color(crop, min_fraction=0.5)
                                color_display = f"{detected_color} " if detected_color else ""
                                tag_text = f"#{trk_id} {color_display}{display_label}"
                                new_boxes.append((bx1, by1, bx2, by2, tag_text))

                                # Save annotated snapshot
                                snap_filename = f"track_{trk_id}.jpg"
                                snap_path = self.snapshots_dir / snap_filename
                                rel_snap_path = f"snapshots/{self.camera_name}/{snap_filename}".replace("\\", "/")

                                annotated = create_annotated_snapshot(
                                    frame,
                                    [bx1, by1, bx2, by2],
                                    f"{color_display}{display_label}",
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

                                # Insert / Update track in SQLite only if live storage enabled
                                if self.store_live_tracks:
                                    track_rec = {
                                        "id": global_trk_id,
                                        "video": rel_rec_path,
                                        "camera": self.camera_name,
                                        "track_id": int(trk_id),
                                        "label": display_label,
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
                                        "label": display_label,
                                        "color": detected_color or "none",
                                        "timestamp": curr_iso,
                                        "offset_s": offset_s
                                    }
                                    self.stats["latest_snapshot"] = rel_snap_path

                        active_preview_boxes = new_boxes
                    except Exception as detect_err:
                        logger.warning(f"[LIVE] Non-fatal error during frame detection/embedding: {detect_err}")

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


class CameraStreamReceiver:
    """
    Dedicated receiver thread per camera stream.
    Connects to stream URL, reads frames with a timeout watchdog,
    and stores only the single latest frame (dropping older unconsumed frames).
    Maintains dedicated tracker instance (BYTETracker) to prevent cross-camera ID collision.
    """
    def __init__(
        self,
        camera_name: str,
        stream_url: str,
        snapshots_dir: Path,
        recorded_dir: Path,
        read_timeout_s: float = 4.0
    ):
        self.camera_name = camera_name.strip()
        self.stream_url = stream_url.strip()
        self.read_timeout_s = read_timeout_s
        self.snapshots_dir = snapshots_dir / self.camera_name
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)
        self.recorded_dir = recorded_dir
        self.recorded_dir.mkdir(parents=True, exist_ok=True)

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        # Single-slot latest frame buffer (older unconsumed frames are dropped)
        self.latest_raw_frame: Optional[Tuple[np.ndarray, float]] = None
        self.latest_jpeg: Optional[bytes] = None

        # Empirical runtime statistics
        self.last_frame_time: Optional[float] = None  # Laptop receive time
        self.frames_received: int = 0
        self.frames_processed: int = 0
        self.reconnect_count: int = 0
        self.errors: List[str] = []
        self.effective_fps: float = 0.0
        self.lag: float = 0.0
        self.status: str = "initialized"

        # Dedicated tracker per camera (tracker isolation fix)
        self.tracker = create_tracker("bytetrack_tuned")
        self.tracks_seen: set = set()
        self.latest_track: Optional[Dict[str, Any]] = None
        self.writer = None
        self.rec_path = None
        self.rel_rec_path = None

        self._fps_window: List[float] = []

    def start(self):
        """Start the stream receiver thread."""
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run_receiver, daemon=True)
        self.thread.start()
        logger.info(f"[RECEIVER:{self.camera_name}] Started ingest on '{self.stream_url}'")

    def stop(self):
        """Stop the stream receiver thread and finalize recordings."""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)
        self.status = "stopped"
        logger.info(f"[RECEIVER:{self.camera_name}] Stopped ingest")

    def pop_latest_frame(self) -> Optional[Tuple[np.ndarray, float]]:
        """
        Atomically retrieve the single latest frame and its laptop receive time,
        dropping any unconsumed prior frames.
        """
        with self.lock:
            if self.latest_raw_frame is None:
                return None
            frame_data = self.latest_raw_frame
            self.latest_raw_frame = None  # Dropped!
            return frame_data

    def _open_capture(self) -> Optional[cv2.VideoCapture]:
        raw_url = self.stream_url.strip()
        # If no protocol is specified and not a local file path, assume http://
        if not raw_url.startswith(("http://", "https://", "rtsp://", "rtmp://")) and not Path(raw_url).exists():
            raw_url = f"http://{raw_url}"

        candidate_urls = [raw_url]
        if raw_url.startswith(("http://", "https://")):
            clean_base = raw_url.rstrip("/")
            for ep in ["/video", "/mjpeg", "/mjpg", "/videofeed", "/live"]:
                if not clean_base.endswith(ep):
                    candidate_urls.append(f"{clean_base}{ep}")

        for cand in candidate_urls:
            logger.info(f"[RECEIVER:{self.camera_name}] Trying capture URL: {cand}")
            cap = cv2.VideoCapture(cand)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    return cap
            cap.release()
        return None

    def _run_receiver(self):
        self.status = "connecting"
        cap = self._open_capture()
        if cap is None:
            self.status = "error"
            self.errors.append(f"Failed to connect to stream at {self.stream_url}")
            self.running = False
            return

        self.status = "running"
        start_dt = datetime.datetime.now()
        start_iso = start_dt.isoformat()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0 or fps > 120:
            fps = 25.0

        rec_filename = f"{self.camera_name}_{start_dt.strftime('%Y%m%d_%H%M%S')}.mp4"
        self.rec_path = self.recorded_dir / rec_filename
        self.rel_rec_path = f"footage/recorded/{rec_filename}".replace("\\", "/")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(self.rec_path), fourcc, fps, (width, height))

        last_frame_time = time.time()

        try:
            while self.running:
                ret, frame = cap.read()
                now = time.time()

                if not ret or frame is None or frame.size == 0:
                    # If reading local MP4 file (simulation), loop back to start
                    if not self.stream_url.lower().startswith(("http://", "https://", "rtsp://")):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(1.0 / fps)
                        continue

                    # Network stream read failed or timed out
                    if now - last_frame_time > self.read_timeout_s:
                        err_msg = f"Read timeout (> {self.read_timeout_s}s) from {self.stream_url}"
                        logger.warning(f"[RECEIVER:{self.camera_name}] {err_msg}")
                        self.errors.append(err_msg)
                        if len(self.errors) > 10:
                            self.errors = self.errors[-10:]
                        self.reconnect_count += 1
                        cap.release()
                        time.sleep(1.0)
                        cap = self._open_capture()
                        if cap is None:
                            time.sleep(1.0)
                            continue
                        last_frame_time = time.time()
                    else:
                        time.sleep(0.01)
                    continue

                recv_time = now
                last_frame_time = recv_time

                with self.lock:
                    self.latest_raw_frame = (frame, recv_time)
                    self.last_frame_time = recv_time
                    self.frames_received += 1
                    if self.writer:
                        try:
                            self.writer.write(frame)
                        except Exception:
                            pass
        finally:
            if cap:
                cap.release()
            if self.writer:
                self.writer.release()
            if self.rec_path and self.rec_path.exists() and self.rec_path.stat().st_size > 0:
                convert_to_h264(self.rec_path)
            self.status = "stopped"

    def record_processed(
        self,
        now: float,
        lag: float,
        tracked_results: List[Dict[str, Any]],
        frame: np.ndarray
    ):
        with self.lock:
            self.frames_processed += 1
            self.lag = round(lag, 3)
            self._fps_window.append(now)
            self._fps_window = [t for t in self._fps_window if now - t <= 3.0]
            if len(self._fps_window) > 1:
                duration = self._fps_window[-1] - self._fps_window[0]
                if duration > 0.1:
                    self.effective_fps = round((len(self._fps_window) - 1) / duration, 1)
                else:
                    self.effective_fps = round(len(self._fps_window) / 3.0, 1)
            else:
                self.effective_fps = round(len(self._fps_window) / 3.0, 1)

            # Update live preview JPEG with annotations
            disp = frame.copy()
            for trk in tracked_results:
                b = trk["bbox"]
                tid = trk["track_id"]
                lbl = trk.get("label", "object")
                conf = trk.get("conf", 0.0)
                tag = f"#{tid} {lbl} ({conf:.2f})"
                cv2.rectangle(disp, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
                cv2.putText(
                    disp, tag, (b[0], max(18, b[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
                )
                if tid not in self.tracks_seen:
                    self.tracks_seen.add(tid)
                    self.latest_track = {
                        "id": f"{self.camera_name}_trk_{tid}",
                        "label": lbl,
                        "timestamp": datetime.datetime.fromtimestamp(now).isoformat()
                    }
            _, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 55])
            self.latest_jpeg = buf.tobytes()

    def get_status(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "camera": self.camera_name,
                "running": self.running,
                "effective_fps": self.effective_fps,
                "lag": self.lag,
                "last_frame_time": self.last_frame_time,
                "errors": list(self.errors),
                "reconnect_count": self.reconnect_count,
                "frames_received": self.frames_received,
                "frames_processed": self.frames_processed,
                "stream_url": self.stream_url,
                "tracks_indexed": len(self.tracks_seen),
                "latest_track": self.latest_track,
                "recorded_file": self.rel_rec_path
            }


class LiveStreamManager:
    """
    Multi-camera Live Streaming Manager.
    - Connects to N streams concurrently with (camera_name, url).
    - Uses one shared YOLO-World detector on GPU protected by a GPU lock.
    - Enforces separate tracker state and ID space per camera (BYTETracker).
    - Always processes the latest frame per camera, immediately dropping older unconsumed frames.
    - Records laptop receive time for accurate timestamping and latency/lag profiling.
    """
    def __init__(
        self,
        yolo_model=None,
        embedder=None,
        classes: Optional[List[str]] = None,
        db_path: str | Path = "index_base/index.db",
        snapshots_dir: str | Path = "index_base/snapshots",
        recorded_dir: str | Path = "footage/recorded",
        conf_thresh: float = 0.15,
        imgsz: int = 640
    ):
        self.yolo = yolo_model
        self.embedder = embedder
        self.classes = classes or []
        self.db_path = Path(db_path)
        self.snapshots_dir = Path(snapshots_dir)
        self.recorded_dir = Path(recorded_dir)
        self.conf_thresh = conf_thresh
        self.imgsz = imgsz

        self.gpu_lock = threading.Lock()
        self.cameras: Dict[str, CameraStreamReceiver] = {}
        self.is_running = False
        self.processor_thread: Optional[threading.Thread] = None

    def start_stream(self, camera: str, url: str) -> Dict[str, Any]:
        """Start or restart a stream for a given camera identifier."""
        camera = camera.strip()
        url = url.strip()
        if not camera or not url:
            raise ValueError("Both camera and url are required.")

        # If camera stream is already running:
        if camera in self.cameras and self.cameras[camera].running:
            # If the user supplied the same camera name but a DIFFERENT URL,
            # they likely added a second phone without editing the camera name box.
            if self.cameras[camera].stream_url != url:
                idx = 2
                while f"mobile_cam{idx:02d}" in self.cameras and self.cameras[f"mobile_cam{idx:02d}"].running:
                    idx += 1
                new_camera = f"mobile_cam{idx:02d}"
                logger.info(f"[MANAGER] '{camera}' already streaming on '{self.cameras[camera].stream_url}'. Assigning '{new_camera}' for '{url}'")
                camera = new_camera
            else:
                self.stop_stream(camera)

        receiver = CameraStreamReceiver(
            camera_name=camera,
            stream_url=url,
            snapshots_dir=self.snapshots_dir,
            recorded_dir=self.recorded_dir,
            read_timeout_s=4.0
        )
        self.cameras[camera] = receiver
        receiver.start()

        # Ensure manager central processing loop is running
        if not self.is_running or self.processor_thread is None or not self.processor_thread.is_alive():
            self.is_running = True
            self.processor_thread = threading.Thread(target=self._process_loop, daemon=True)
            self.processor_thread.start()

        logger.info(f"[MANAGER] Started stream for '{camera}' at '{url}'")
        return {"status": "started", "camera": camera, "url": url}

    def stop_stream(self, camera: str) -> Dict[str, Any]:
        """Stop streaming for a specific camera."""
        camera = camera.strip()
        if camera in self.cameras:
            receiver = self.cameras[camera]
            receiver.stop()
            stats = receiver.get_status()
            logger.info(f"[MANAGER] Stopped stream for '{camera}'")
            return {"status": "stopped", "camera": camera, "stats": stats}
        return {"status": "not_found", "camera": camera}

    def get_status(self) -> Dict[str, Any]:
        """Retrieve status listing for all cameras."""
        cams_status = {}
        for cam_name, receiver in self.cameras.items():
            cams_status[cam_name] = receiver.get_status()

        active_cams = [r for r in self.cameras.values() if r.running]
        primary_cam = active_cams[0] if active_cams else (list(self.cameras.values())[0] if self.cameras else None)

        status_dict = {
            "cameras": cams_status,
            "total_active": len(active_cams),
            "status": "streaming" if active_cams else "idle"
        }
        if primary_cam:
            primary_stats = primary_cam.get_status()
            status_dict["camera"] = primary_stats["camera"]
            status_dict["effective_fps"] = primary_stats["effective_fps"]
            status_dict["lag"] = primary_stats["lag"]
            status_dict["last_frame_time"] = primary_stats["last_frame_time"]
            status_dict["frames_read"] = primary_stats["frames_received"]
            status_dict["tracks_indexed"] = primary_stats["tracks_indexed"]
            status_dict["latest_track"] = primary_stats["latest_track"]
            status_dict["recorded_file"] = primary_stats["recorded_file"]
            status_dict["errors"] = primary_stats["errors"]
            status_dict["reconnect_count"] = primary_stats["reconnect_count"]
        return status_dict

    def extend_vocabulary(self, phrase: str) -> float:
        """Dynamically add vocabulary term to shared YOLO model."""
        phrase_clean = phrase.strip().lower()
        if not phrase_clean or phrase_clean in self.classes:
            return 0.0

        with self.gpu_lock:
            t0 = time.perf_counter()
            self.classes.append(phrase_clean)
            try:
                if self.yolo is not None:
                    if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "clip_model"):
                        self.yolo.model.clip_model = None
                    if hasattr(self.yolo, "model") and hasattr(self.yolo.model, "float"):
                        self.yolo.model.float()
                    self.yolo.set_classes(self.classes)
                delay_ms = (time.perf_counter() - t0) * 1000.0
                logger.info(f"[MANAGER] Extended vocab with '{phrase_clean}' in {delay_ms:.2f} ms")
                return delay_ms
            except Exception as e:
                logger.warning(f"[MANAGER] Extend vocab for '{phrase_clean}' failed: {e}")
                if phrase_clean in self.classes:
                    self.classes.remove(phrase_clean)
                return 0.0

    def _process_loop(self):
        """
        Central processing worker thread.
        Pulls latest frame from active cameras, runs shared GPU detector under lock,
        and executes camera-isolated tracking.
        """
        while self.is_running:
            active_receivers = [r for r in list(self.cameras.values()) if r.running]
            if not active_receivers:
                time.sleep(0.05)
                continue

            processed_any = False
            for receiver in active_receivers:
                frame_data = receiver.pop_latest_frame()
                if frame_data is None:
                    continue

                frame, recv_time = frame_data
                processed_any = True
                h, w = frame.shape[:2]

                with self.gpu_lock:
                    dets = []
                    if self.yolo is not None:
                        try:
                            results = self.yolo.predict(
                                frame,
                                conf=self.conf_thresh,
                                imgsz=self.imgsz,
                                verbose=False
                            )
                            if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
                                boxes = results[0].boxes
                                cls_ids = boxes.cls.cpu().numpy().astype(int)
                                confs = boxes.conf.cpu().numpy().astype(float)
                                xyxy = boxes.xyxy.cpu().numpy().astype(int)
                                for c_id, conf, box in zip(cls_ids, confs, xyxy):
                                    label = self.classes[c_id] if c_id < len(self.classes) else "object"
                                    dets.append({
                                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                                        "conf": float(conf),
                                        "label": label
                                    })
                        except Exception as e:
                            logger.warning(f"[MANAGER:{receiver.camera_name}] Detection error: {e}")

                    # Run isolated tracker for this camera
                    try:
                        tracked_dets = run_tracker_on_detections(
                            receiver.tracker,
                            dets,
                            frame,
                            w,
                            h
                        )
                    except Exception as e:
                        logger.warning(f"[MANAGER:{receiver.camera_name}] Tracking error: {e}")
                        tracked_dets = []

                now = time.time()
                lag = now - recv_time
                receiver.record_processed(now, lag, tracked_dets, frame)

            if not processed_any:
                time.sleep(0.005)

