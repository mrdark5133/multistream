"""
Simulated Mobile Phone MJPEG Streamer (High-Performance In-Memory).
Preloads frames from test videos into memory to guarantee seamless, zero-stutter
25 FPS network streaming simulating Android IP Webcam / iOS camera streams.
"""

import cv2
import time
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

VIDEOS = {
    "cam1": Path("footage/test_video01.mp4"),
    "cam2": Path("footage/test_video02.mp4"),
    "cam3": Path("footage/traffic_video.mp4"),
    "cam4": Path("footage/test_video03.mp4"),
}

class InMemoryStreamer:
    def __init__(self, video_path: Path, target_fps: float = 25.0):
        self.video_path = video_path
        self.target_fps = target_fps
        self.interval = 1.0 / target_fps
        self.frames = []
        self._preload()
        self.current_idx = 0
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _preload(self):
        cap = cv2.VideoCapture(str(self.video_path))
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            h, w = frame.shape[:2]
            if w > 640:
                frame = cv2.resize(frame, (640, int(h * 640 / w)))
            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            self.frames.append(buf.tobytes())
            if len(self.frames) >= 300:
                break
        cap.release()
        if not self.frames:
            import numpy as np
            blank = cv2.imencode('.jpg', np.zeros((480, 640, 3), dtype=np.uint8))[1].tobytes()
            self.frames.append(blank)

    def _loop(self):
        while self.running:
            t0 = time.time()
            with self.lock:
                self.current_idx = (self.current_idx + 1) % len(self.frames)
            elapsed = time.time() - t0
            sleep_t = max(0.001, self.interval - elapsed)
            time.sleep(sleep_t)

    def get_frame(self):
        with self.lock:
            return self.frames[self.current_idx]

STREAMERS = {}

def get_streamer(key: str) -> InMemoryStreamer:
    if key not in STREAMERS:
        video_path = VIDEOS.get(key, VIDEOS["cam1"])
        if not video_path.exists():
            video_path = Path("footage/traffic_video.mp4")
        STREAMERS[key] = InMemoryStreamer(video_path)
    return STREAMERS[key]

class MJPEGHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        path = self.path.split('?')[0].strip('/')
        port = self.server.server_port
        key = "cam1"
        if "cam2" in path or port == 8082:
            key = "cam2"
        elif "cam3" in path or port == 8083:
            key = "cam3"
        elif "cam4" in path or port == 8084:
            key = "cam4"
        elif "cam1" in path or port == 8081:
            key = "cam1"

        streamer = get_streamer(key)

        try:
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.end_headers()
            while True:
                jpeg = streamer.get_frame()
                if jpeg:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                time.sleep(streamer.interval)
        except Exception:
            pass

def run_server(port: int):
    server = ThreadingHTTPServer(('0.0.0.0', port), MJPEGHandler)
    server.serve_forever()

if __name__ == "__main__":
    import numpy as np
    print("[SIMULATOR] Starting high-performance simulated phone MJPEG servers...")
    get_streamer("cam1")
    get_streamer("cam2")
    get_streamer("cam3")
    get_streamer("cam4")
    
    t1 = threading.Thread(target=run_server, args=(8081,), daemon=True)
    t2 = threading.Thread(target=run_server, args=(8082,), daemon=True)
    t3 = threading.Thread(target=run_server, args=(8083,), daemon=True)
    t4 = threading.Thread(target=run_server, args=(8084,), daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    t4.start()
    
    print("[SIMULATOR] Ports 8081, 8082, 8083, 8084 online.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
