import urllib.request
from pathlib import Path
import sys

WEIGHTS_CONFIG = [
    {
        "name": "Piper Voice Model (en_US-lessac-low.onnx)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx",
        "dest": Path("weights/piper/en/en_US/lessac/low/en_US-lessac-low.onnx")
    },
    {
        "name": "Piper Voice Config (en_US-lessac-low.onnx.json)",
        "url": "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0/en/en_US/lessac/low/en_US-lessac-low.onnx.json",
        "dest": Path("weights/piper/en/en_US/lessac/low/en_US-lessac-low.onnx.json")
    }
]

def download_weights():
    root = Path(__file__).resolve().parent.parent
    print("=== MULTIStream Weight Downloader ===")
    for item in WEIGHTS_CONFIG:
        dest_path = root / item["dest"]
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        if dest_path.exists() and dest_path.stat().st_size > 0:
            print(f"[OK] Already exists: {item['dest']}")
            continue
        print(f"[DOWNLOADING] {item['name']}...")
        print(f"  URL: {item['url']}")
        try:
            urllib.request.urlretrieve(item["url"], str(dest_path))
            print(f"  -> Saved to {dest_path} ({dest_path.stat().st_size / (1024*1024):.2f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {item['name']}: {e}")
            sys.exit(1)

    print("\nAll required offline model weights are downloaded and ready!")

if __name__ == "__main__":
    download_weights()
