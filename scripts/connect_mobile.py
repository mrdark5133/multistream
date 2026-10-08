import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.ingest.detect_track import load_yolo_world
from src.ingest.embed import SigLIPEmbedder
from src.ingest.live_stream import LiveStreamWorker


def main():
    parser = argparse.ArgumentParser(description="MULTIStream Mobile Phone / IP Webcam Live Stream Connector")
    parser.add_argument("--url", type=str, default=None, help="Live stream URL (e.g. http://192.168.1.100:8080/video or rtsp://...)")
    parser.add_argument("--camera", type=str, default="mobile_cam01", help="Camera name identifier (default: mobile_cam01)")
    parser.add_argument("--demo", action="store_true", help="Run simulated live stream using footage/traffic_video.mp4")
    parser.add_argument("--duration", type=int, default=30, help="Run duration in seconds (default: 30s)")
    args = parser.parse_args()

    stream_url = args.url
    if args.demo or not stream_url:
        demo_file = ROOT_DIR / "footage" / "traffic_video.mp4"
        if not demo_file.exists():
            demo_file = ROOT_DIR / "footage" / "test_video01.mp4"
        stream_url = str(demo_file)
        print(f"[*] Running in DEMO mode streaming from: {stream_url}")
    else:
        print(f"[*] Connecting to live phone stream at: {stream_url}")

    print(f"[*] Camera Name: {args.camera}")
    print("[*] Loading YOLO-World and SigLIP models...")

    yolo, classes = load_yolo_world(
        weights_path="yolov8s-worldv2.pt",
        vocab_path="config/vocab.yaml",
        device="cuda:0"
    )
    embedder = SigLIPEmbedder(device="cuda:0")

    worker = LiveStreamWorker(
        stream_url=stream_url,
        camera_name=args.camera,
        db_path="index_base/index.db",
        stride=5
    )

    print("[*] Starting live real-time ingestion, color extraction, and database indexing...")
    worker.start(yolo_model=yolo, embedder=embedder, classes=classes)

    start_t = time.time()
    try:
        while time.time() - start_t < args.duration:
            stats = worker.stats
            latest = stats.get("latest_track") or {}
            color_txt = f"[{latest.get('color', 'unknown')}]" if latest else ""
            label_txt = latest.get("label", "none")
            sys.stdout.write(
                f"\r[LIVE] Status: {stats['status']} | FPS: {stats['fps']} | "
                f"Frames: {stats['frames_read']} | Tracks Indexed: {stats['tracks_indexed']} | "
                f"Latest: {color_txt} {label_txt}      "
            )
            sys.stdout.flush()
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[*] Stopping stream...")
    finally:
        worker.stop()

    print("\n" + "=" * 80)
    print("LIVE MOBILE INGESTION FINISHED")
    print(f"Camera:         {args.camera}")
    print(f"Tracks Indexed: {worker.stats['tracks_indexed']}")
    print(f"Frames Read:    {worker.stats['frames_read']}")
    print(f"Ready to query in chat: 'what did you see in {args.camera}?'")
    print("=" * 80)


if __name__ == "__main__":
    main()
