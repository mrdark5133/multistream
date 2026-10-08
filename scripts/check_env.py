"""
Environment and Hardware Diagnostic Script for MULTIStream
Checks:
- OS & Architecture
- Python version & executable
- Git version
- FFmpeg & FFprobe versions
- CUDA availability, device name, VRAM (total, allocated, free via nvidia-smi)
- Core package versions (torch, torchvision, ultralytics, transformers)
- LLM provider status (present/missing, minimal test call if key present)
"""

import os
import sys
import time
import subprocess
import platform
from pathlib import Path
from dotenv import load_dotenv

# Load .env if present
load_dotenv(override=True)

def print_section(title: str):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def run_cmd(cmd: list[str]) -> tuple[int, str]:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10)
        return res.returncode, (res.stdout + res.stderr).strip()
    except Exception as e:
        return -1, str(e)

def main():
    print_section("1. SYSTEM & OPERATING SYSTEM")
    print(f"OS: {platform.system()} {platform.release()} (Version: {platform.version()})")
    print(f"Architecture: {platform.machine()}")
    print(f"Python: {platform.python_version()} ({sys.executable})")

    print_section("2. SYSTEM TOOLS")
    code, out = run_cmd(["git", "--version"])
    print(f"Git: {out if code == 0 else 'NOT FOUND'}")

    code, out = run_cmd(["ffmpeg", "-version"])
    ffmpeg_line = out.split("\n")[0] if code == 0 else "NOT FOUND / ERROR: " + out
    print(f"FFmpeg: {ffmpeg_line}")

    code, out = run_cmd(["ffprobe", "-version"])
    ffprobe_line = out.split("\n")[0] if code == 0 else "NOT FOUND / ERROR: " + out
    print(f"FFprobe: {ffprobe_line}")

    print_section("3. NVIDIA & CUDA HARDWARE")
    code, smi_out = run_cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version", "--format=csv,noheader"])
    if code == 0:
        gpu_info = [x.strip() for x in smi_out.split(",")]
        if len(gpu_info) >= 5:
            print(f"GPU Name:       {gpu_info[0]}")
            print(f"Total VRAM:     {gpu_info[1]}")
            print(f"Used VRAM:      {gpu_info[2]}")
            print(f"Free VRAM:      {gpu_info[3]}")
            print(f"Driver Version: {gpu_info[4]}")
        else:
            print(f"nvidia-smi: {smi_out}")
    else:
        print(f"nvidia-smi: NOT AVAILABLE ({smi_out})")

    print_section("4. PYTORCH & CUDA ACCELERATION")
    try:
        import torch
        print(f"PyTorch version:       {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA Available:        {cuda_avail}")
        if cuda_avail:
            print(f"CUDA Device Count:     {torch.cuda.device_count()}")
            print(f"CUDA Device Name:      {torch.cuda.get_device_name(0)}")
            print(f"CUDA Device Capability: {torch.cuda.get_device_capability(0)}")
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            print(f"Torch Free VRAM:       {free_mem / (1024**2):.1f} MB / {total_mem / (1024**2):.1f} MB")
        else:
            print("WARNING: CUDA is FALSE. Hardware acceleration not active in PyTorch!")
    except ImportError:
        print("PyTorch is NOT installed in this environment.")

    print_section("5. CORE ML PACKAGES")
    for pkg in ["torchvision", "ultralytics", "transformers", "accelerate", "fastapi"]:
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            print(f"{pkg:16}: {ver}")
        except ImportError:
            print(f"{pkg:16}: NOT INSTALLED")

    print_section("6. LLM PROVIDERS STATUS")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    claude_model = os.environ.get("CLAUDE_MODEL", "claude-3-5-sonnet-20241022").strip()
    gemini_model = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash").strip()

    print(f"ANTHROPIC_API_KEY: {'PRESENT' if anthropic_key else 'MISSING'}")
    print(f"GEMINI_API_KEY:    {'PRESENT' if gemini_key else 'MISSING'}")

    if anthropic_key:
        print("\nTesting Anthropic API connection...")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_key)
            t0 = time.perf_counter()
            resp = client.messages.create(
                model=claude_model,
                max_tokens=20,
                messages=[{"role": "user", "content": "Reply with 'OK'"}]
            )
            elapsed = (time.perf_counter() - t0) * 1000
            content = resp.content[0].text if resp.content else ""
            print(f"Anthropic Success: model={claude_model}, latency={elapsed:.1f}ms, response={content.strip()!r}")
        except Exception as e:
            print(f"Anthropic Call Failed: {type(e).__name__}: {e}")
    else:
        print("Anthropic: NOT RUN (Key missing)")

    if gemini_key:
        print("\nTesting Gemini API connection...")
        try:
            # Try new google-genai SDK first, then google-generativeai fallback
            try:
                from google import genai
                client = genai.Client(api_key=gemini_key)
                t0 = time.perf_counter()
                resp = client.models.generate_content(
                    model=gemini_model,
                    contents="Reply with 'OK'",
                )
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"Gemini Success: model={gemini_model}, latency={elapsed:.1f}ms, response={resp.text.strip()!r}")
            except ImportError:
                import google.generativeai as genai
                genai.configure(api_key=gemini_key)
                t0 = time.perf_counter()
                model = genai.GenerativeModel(gemini_model)
                resp = model.generate_content("Reply with 'OK'")
                elapsed = (time.perf_counter() - t0) * 1000
                print(f"Gemini Success: model={gemini_model}, latency={elapsed:.1f}ms, response={resp.text.strip()!r}")
        except Exception as e:
            print(f"Gemini Call Failed: {type(e).__name__}: {e}")
    else:
        print("Gemini: NOT RUN (Key missing)")

    print_section("DIAGNOSTIC SUMMARY COMPLETE")

if __name__ == "__main__":
    main()
