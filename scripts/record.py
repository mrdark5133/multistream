import argparse
import sys
import os
import subprocess
import time
import datetime
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [RECORD] %(message)s"
)
logger = logging.getLogger("multistream.record")


def record_stream(
    stream_url: str,
    camera_name: str,
    output_dir: Path,
    segment_duration_s: int = 10,
    max_duration_s: int | None = None,
    is_simulated: bool = False
) -> list[Path]:
    """
    Record an RTSP / HTTP video stream into segmented MP4 files using ffmpeg.
    If is_simulated is True, uses '-re' flag to simulate live real-time rate from a file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp_prefix = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_pattern = str(output_dir / f"{camera_name}_{timestamp_prefix}_%03d.mp4")

    # Resolve ffmpeg executable
    ffmpeg_exe = os.environ.get("FFMPEG_BIN", "ffmpeg")
    winget_path = Path(r"C:\Users\Harivarman R\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe")
    if winget_path.exists() and (ffmpeg_exe == "ffmpeg" or not os.path.exists(ffmpeg_exe)):
        ffmpeg_exe = str(winget_path)

    # Construct ffmpeg command
    cmd = [ffmpeg_exe, "-y"]

    if is_simulated:
        cmd.extend(["-re"])  # Read input at native frame rate to simulate live streaming

    if stream_url.lower().startswith("rtsp://"):
        cmd.extend(["-rtsp_transport", "tcp"])

    cmd.extend(["-i", stream_url])

    if max_duration_s:
        cmd.extend(["-t", str(max_duration_s)])

    cmd.extend([
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        "-f", "segment",
        "-segment_time", str(segment_duration_s),
        "-reset_timestamps", "1",
        out_pattern
    ])

    mode_label = "SIMULATED STREAM (Local File as RTSP Stream)" if is_simulated else "LIVE STREAM"
    logger.info(f"Starting recording [{mode_label}] for camera '{camera_name}'...")
    logger.info(f"Source URL:       {stream_url}")
    logger.info(f"Segment Duration: {segment_duration_s}s")
    logger.info(f"Target Output:    {out_pattern}")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = proc.communicate()

    # Find created segment files
    created_segments = sorted(output_dir.glob(f"{camera_name}_{timestamp_prefix}_*.mp4"))
    logger.info(f"Recording complete. Created {len(created_segments)} segment files:")
    for seg in created_segments:
        logger.info(f"  -> {seg.name} ({seg.stat().st_size / 1024:.1f} KB)")

    return created_segments


def main():
    parser = argparse.ArgumentParser(description="MULTIStream RTSP / Live Video Stream Recorder")
    parser.add_argument("--stream", type=str, required=True, help="RTSP stream URL or path to video file for simulated stream")
    parser.add_argument("--camera", type=str, default="cam_stream1", help="Camera identifier")
    parser.add_argument("--output-dir", type=str, default="footage/recorded", help="Directory to save recorded stream chunks")
    parser.add_argument("--segment-duration", type=int, default=5, help="Duration in seconds per video chunk")
    parser.add_argument("--max-duration", type=int, default=10, help="Total recording duration limit in seconds")
    parser.add_argument("--simulated", action="store_true", help="Explicitly mark stream as simulated from file")
    args = parser.parse_args()

    stream_path = args.stream
    is_simulated = args.simulated or not (stream_path.lower().startswith("rtsp://") or stream_path.lower().startswith("http://"))

    record_stream(
        stream_url=stream_path,
        camera_name=args.camera,
        output_dir=Path(args.output_dir).resolve(),
        segment_duration_s=args.segment_duration,
        max_duration_s=args.max_duration,
        is_simulated=is_simulated
    )


if __name__ == "__main__":
    main()
