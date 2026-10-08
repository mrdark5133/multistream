"""
Phase 0 Fix-up Task 2: Clean measurement of Condition (b)
5 nvidia-smi readings 2s apart + torch.cuda.mem_get_info() + process breakdown
"""
import time
import subprocess
import torch

def get_smi_line():
    res = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    if res.returncode == 0:
        tot, used, free = [float(x.strip()) for x in res.stdout.strip().split(",")]
        return tot, used, free
    return None, None, None

def main():
    print("=" * 60)
    print("PHASE 0 TASK 2: Clean Condition (b) VRAM Measurements")
    print("=" * 60)

    print("\n1. 5 Consecutive nvidia-smi readings (2 seconds apart):")
    readings = []
    for i in range(1, 6):
        tot, used, free = get_smi_line()
        readings.append((tot, used, free))
        print(f"   Reading {i}: Total={tot:.1f} MiB | Used={used:.1f} MiB | Free={free:.1f} MiB")
        if i < 5:
            time.sleep(2)

    avg_used = sum(r[1] for r in readings) / len(readings)
    avg_free = sum(r[2] for r in readings) / len(readings)
    print(f"\n   Average over 5 readings: Used={avg_used:.1f} MiB | Free={avg_free:.1f} MiB")

    print("\n2. PyTorch CUDA Memory Query (torch.cuda.mem_get_info):")
    if torch.cuda.is_available():
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_mb = free_bytes / (1024**2)
        total_mb = total_bytes / (1024**2)
        allocated_mb = torch.cuda.memory_allocated(0) / (1024**2)
        reserved_mb = torch.cuda.memory_reserved(0) / (1024**2)

        print(f"   torch.cuda.mem_get_info free : {free_mb:.2f} MB ({free_bytes} bytes)")
        print(f"   torch.cuda.mem_get_info total: {total_mb:.2f} MB ({total_bytes} bytes)")
        print(f"   torch.cuda.memory_allocated : {allocated_mb:.2f} MB")
        print(f"   torch.cuda.memory_reserved  : {reserved_mb:.2f} MB")

        # Explain the gap
        tot_smi, used_smi, free_smi = get_smi_line()
        print("\n3. Gap Analysis Between nvidia-smi and PyTorch:")
        print(f"   nvidia-smi reports physical GPU memory: Total={tot_smi:.1f} MiB, Used={used_smi:.1f} MiB, Free={free_smi:.1f} MiB")
        print(f"   torch.cuda.mem_get_info() reports: Free={free_mb:.1f} MB, Total={total_mb:.1f} MB")
        gap = free_mb - free_smi
        print(f"   Discrepancy (PyTorch free - nvidia-smi free): {gap:+.1f} MB")
        print("   Explanation:")
        print("   On Windows (WDDM 3.x driver model), nvidia-smi reports dedicated on-board VRAM used by all desktop processes.")
        print("   In contrast, cudaMemGetInfo on WDDM reports the memory budget available to the current DirectX/CUDA process,")
        print("   which accounts for dynamic OS paging/virtual memory overcommit buffers managed by the Windows GPU scheduler.")
    else:
        print("   CUDA not available in PyTorch.")

    print("\n4. Active GPU Processes (from nvidia-smi):")
    res = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print(res.stdout.strip())

if __name__ == "__main__":
    main()
