"""
Multi-Camera Live Ingest Benchmark Script.
Empirical evaluation with 1, 2, and 3 simulated phone streams.
Measures:
  - Effective FPS per stream
  - Latency / Lag (seconds) per stream (using laptop receive time)
  - Peak VRAM (MiB)
  - Process CPU (%)
All results labeled SIMULATED.
"""

import time
import json
import urllib.request
import psutil
import torch
from typing import Dict, Any, List

API_BASE = "http://127.0.0.1:8000"

STREAMS = [
    {"camera": "sim_cam01", "url": "http://127.0.0.1:8081/video"},
    {"camera": "sim_cam02", "url": "http://127.0.0.1:8082/video"},
    {"camera": "sim_cam03", "url": "http://127.0.0.1:8083/video"}
]

def api_post(endpoint: str, payload: dict) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{API_BASE}{endpoint}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def api_get(endpoint: str) -> dict:
    req = urllib.request.Request(f"{API_BASE}{endpoint}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def stop_all_active_streams():
    status = api_get("/stream/status")
    for cam_name, cam_info in status.get("cameras", {}).items():
        if cam_info.get("running"):
            try:
                api_post("/stream/stop", {"camera": cam_name, "auto_ingest": False})
            except Exception:
                pass
    time.sleep(1.0)

def get_vram_and_cpu(process: psutil.Process):
    cpu = psutil.cpu_percent(interval=0.05)
    vram_mb = 0.0
    try:
        from src.utils.vram import get_nvml_vram_info
        total, used, free = get_nvml_vram_info()
        vram_mb = float(used)
    except Exception:
        try:
            vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        except Exception:
            pass
    return cpu, vram_mb

def run_test_condition(n_streams: int, duration_s: float = 16.0) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f" [SIMULATED] STARTING TEST: {n_streams} SIMULATED STREAMS")
    print(f"=======================================================")
    stop_all_active_streams()

    # Find uvicorn server process
    server_proc = None
    current_pid = psutil.Process().pid
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = " ".join(proc.info['cmdline'] or [])
            if "uvicorn" in cmd and "src.api.app:app" in cmd:
                server_proc = proc
                break
        except Exception:
            pass
    if server_proc is None:
        server_proc = psutil.Process()

    # Warm-up cpu measurement
    server_proc.cpu_percent(interval=None)

    # Start streams
    active_configs = STREAMS[:n_streams]
    for cfg in active_configs:
        res = api_post("/stream/start", {"camera": cfg["camera"], "url": cfg["url"]})
        print(f" [SIMULATED] Started {cfg['camera']} -> {cfg['url']}")
        time.sleep(0.5)

    print(f" [SIMULATED] Warming up & collecting metrics for {duration_s} seconds...")
    samples = []
    start_t = time.time()
    
    # Warm up for 4 seconds to let FPS windows settle
    time.sleep(4.0)
    
    sample_interval = 1.0
    while time.time() - start_t < duration_s:
        cpu, vram_mb = get_vram_and_cpu(server_proc)
        status = api_get("/stream/status")
        cams_status = status.get("cameras", {})
        sample = {
            "timestamp": time.time(),
            "cpu_percent": cpu,
            "vram_mb": vram_mb,
            "cameras": {
                cam_name: {
                    "fps": cams_status[cam_name].get("effective_fps", 0.0),
                    "lag": cams_status[cam_name].get("lag", 0.0),
                    "frames_recvd": cams_status[cam_name].get("frames_received", 0),
                    "frames_proc": cams_status[cam_name].get("frames_processed", 0),
                    "reconnects": cams_status[cam_name].get("reconnect_count", 0),
                    "errors": cams_status[cam_name].get("errors", [])
                }
                for cam_name in [c["camera"] for c in active_configs]
                if cam_name in cams_status
            }
        }
        samples.append(sample)
        time.sleep(sample_interval)

    # Stop all streams
    for cfg in active_configs:
        api_post("/stream/stop", {"camera": cfg["camera"], "auto_ingest": False})
        print(f" [SIMULATED] Stopped {cfg['camera']}")

    # Compute aggregate stats
    if not samples:
        return {"error": "No samples recorded"}

    peak_vram = max(s["vram_mb"] for s in samples)
    avg_cpu = sum(s["cpu_percent"] for s in samples) / len(samples)

    cam_stats = {}
    for cfg in active_configs:
        cam_id = cfg["camera"]
        fps_list = [s["cameras"][cam_id]["fps"] for s in samples if cam_id in s["cameras"]]
        lag_list = [s["cameras"][cam_id]["lag"] for s in samples if cam_id in s["cameras"]]
        last_recvd = samples[-1]["cameras"][cam_id]["frames_recvd"] if cam_id in samples[-1]["cameras"] else 0
        last_proc = samples[-1]["cameras"][cam_id]["frames_proc"] if cam_id in samples[-1]["cameras"] else 0
        reconnects = samples[-1]["cameras"][cam_id]["reconnects"] if cam_id in samples[-1]["cameras"] else 0
        errors = samples[-1]["cameras"][cam_id]["errors"] if cam_id in samples[-1]["cameras"] else []

        avg_fps = sum(fps_list) / len(fps_list) if fps_list else 0.0
        avg_lag = sum(lag_list) / len(lag_list) if lag_list else 0.0
        cam_stats[cam_id] = {
            "avg_fps": round(avg_fps, 2),
            "max_fps": round(max(fps_list) if fps_list else 0.0, 2),
            "min_fps": round(min(fps_list) if fps_list else 0.0, 2),
            "avg_lag_s": round(avg_lag, 3),
            "max_lag_s": round(max(lag_list) if lag_list else 0.0, 3),
            "min_lag_s": round(min(lag_list) if lag_list else 0.0, 3),
            "frames_received": last_recvd,
            "frames_processed": last_proc,
            "frames_dropped": max(0, last_recvd - last_proc),
            "reconnect_count": reconnects,
            "errors": errors
        }

    total_fps = sum(cs["avg_fps"] for cs in cam_stats.values())

    result = {
        "n_streams": n_streams,
        "label": "SIMULATED",
        "cam_stats": cam_stats,
        "total_system_fps": round(total_fps, 2),
        "avg_cpu_percent": round(avg_cpu, 1),
        "peak_vram_mib": round(peak_vram, 1),
        "num_samples": len(samples)
    }
    return result

def main():
    print("[SIMULATED] MULTI-PHONE LIVE INGEST BENCHMARK STARTING...")
    print("[SIMULATED] Environment: Laptop GPU (RTX 4060 Laptop GPU 8GB), YOLO-World v2 + BYTETracker per cam.")
    
    results = {}
    for n in [1, 2, 3]:
        res = run_test_condition(n, duration_s=16.0)
        results[n] = res
        time.sleep(2.0)

    print("\n\n=======================================================")
    print(" [SIMULATED] FINAL BENCHMARK SUMMARY (RAW EMPIRICAL)")
    print("=======================================================")
    print(json.dumps(results, indent=2))

    with open("footage/simulated_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to footage/simulated_benchmark_results.json")

if __name__ == "__main__":
    main()
