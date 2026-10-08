import cv2
from pathlib import Path

out_dir = Path("footage/orientation_check")
out_dir.mkdir(parents=True, exist_ok=True)

clips = [
    "footage/test_video01.mp4",
    "footage/test_video02.mp4",
    "footage/test_video03.mp4"
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
        print(f"Decoded {clip_path}: shape=({w}x{h}), auto_rotate_prop={auto_rotate}, saved={save_path}")
    else:
        print(f"Failed to decode {clip_path}")
    cap.release()
