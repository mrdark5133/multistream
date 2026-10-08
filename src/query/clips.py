import os
import subprocess
from pathlib import Path
from typing import Optional


def extract_video_clip(
    video_path: str | Path,
    output_path: str | Path,
    start_s: float,
    end_s: float,
    padding_s: float = 2.0,
    ffmpeg_exe: Optional[str] = None
) -> str:
    """
    Extract a snippet from source video on-demand using ffmpeg with boundary padding.
    Returns the string path of the extracted video clip.
    """
    v_path = Path(video_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ffmpeg_exe is None:
        ffmpeg_exe = os.environ.get("FFMPEG_BIN", "ffmpeg")
        # Check standard winget fallback
        winget_path = Path(r"C:\Users\Harivarman R\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-9.0.2-full_build\bin\ffmpeg.exe")
        if winget_path.exists():
            ffmpeg_exe = str(winget_path)

    # Apply boundary padding
    clip_start = max(0.0, start_s - padding_s)
    duration = max(1.0, (end_s - start_s) + 2 * padding_s)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-ss", f"{clip_start:.3f}",
        "-i", str(v_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-an",
        str(out_path)
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return str(out_path)
    except Exception as e:
        print(f"Error during ffmpeg clip extraction: {e}")
        return ""
