import cv2
from pathlib import Path

print(f"OpenCV version: {cv2.__version__}")
out_dir = Path("footage/orientation_check")
out_dir.mkdir(parents=True, exist_ok=True)

clips = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4",
    "footage/test_landscape.mp4",
    "footage/test_landscape2.mp4"
]

for clip_path in clips:
    cap = cv2.VideoCapture(clip_path)
    # Check orientation auto property
    auto_rotate = cap.get(cv2.CAP_PROP_ORIENTATION_AUTO)
    # Read frame at ~1.0s (frame 30)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ret, frame = cap.read()
    if ret:
        name = Path(clip_path).stem
        save_path = out_dir / f"frame_{name}.jpg"
        cv2.imwrite(str(save_path), frame)
        h, w, c = frame.shape
        ff_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ff_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Decoded {clip_path}: frame.shape (H, W)=({h}, {w}) | ffprobe (width, height)=({ff_w}, {ff_h}), auto_rotate_prop={auto_rotate}, saved={save_path}")
    else:
        print(f"Failed to decode {clip_path}")
    cap.release()
