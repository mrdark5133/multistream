"""
Phase 0 Fix-up 2 Task 1: Clean Condition (b) + PyTorch Allocation Stress Test
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
    print("=" * 65)
    print("PHASE 0 TASK 1: Clean Condition (b) Measurements & Allocation Stress")
    print("=" * 65)

    print("\n1. 5 Consecutive nvidia-smi readings (2 seconds apart):")
    readings = []
    for i in range(1, 6):
        tot, used, free = get_smi_line()
        readings.append((tot, used, free))
        print(f"   Reading {i}: Total={tot:.1f} MiB | Used={used:.1f} MiB | Free={free:.1f} MiB")
        if i < 5:
            time.sleep(2)

    avg_tot = sum(r[0] for r in readings) / len(readings)
    avg_used = sum(r[1] for r in readings) / len(readings)
    avg_free = sum(r[2] for r in readings) / len(readings)
    print(f"\n   5-Sample Average: Total={avg_tot:.1f} MiB | Used={avg_used:.1f} MiB | Free={avg_free:.1f} MiB")

    print("\n2. PyTorch Initial CUDA Memory Query (torch.cuda.mem_get_info):")
    if not torch.cuda.is_available():
        print("   CUDA not available in PyTorch.")
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    free_mb = free_bytes / (1024**2)
    total_mb = total_bytes / (1024**2)
    allocated_mb = torch.cuda.memory_allocated(0) / (1024**2)
    reserved_mb = torch.cuda.memory_reserved(0) / (1024**2)

    print(f"   torch.cuda.mem_get_info free : {free_mb:.2f} MB ({free_bytes} bytes)")
    print(f"   torch.cuda.mem_get_info total: {total_mb:.2f} MB ({total_bytes} bytes)")
    print(f"   torch.cuda.memory_allocated : {allocated_mb:.2f} MB")
    print(f"   torch.cuda.memory_reserved  : {reserved_mb:.2f} MB")

    print("\n3. Gap Analysis (using 5-sample average):")
    print(f"   nvidia-smi 5-sample avg free: {avg_free:.1f} MiB")
    print(f"   torch.cuda.mem_get_info free: {free_mb:.1f} MB")
    gap = free_mb - avg_free
    print(f"   Gap (PyTorch free - nvidia-smi free): {gap:+.1f} MB")

    print("\n4. Empirical Allocation Stress Test (100 MB steps until CUDA OOM):")
    tensors = []
    step_bytes = 100 * 1024 * 1024 # 100 MB
    elements_per_step = step_bytes // 4 # float32
    step_count = 0

    try:
        while True:
            t = torch.empty((elements_per_step,), dtype=torch.float32, device="cuda:0")
            # Touch memory to ensure actual physical allocation
            t.fill_(1.0)
            tensors.append(t)
            step_count += 1
            cur_alloc = torch.cuda.memory_allocated() / (1024**2)
            _, smi_u, smi_f = get_smi_line()
            print(f"   Allocated step {step_count:2d} (+100 MB) -> Total PyTorch Alloc: {cur_alloc:.1f} MB | SMI Used: {smi_u:.1f} MiB | SMI Free: {smi_f:.1f} MiB")
    except torch.cuda.OutOfMemoryError as e:
        print(f"\n   Hit torch.cuda.OutOfMemoryError at step {step_count + 1}!")
        print(f"   Total successfully allocated in PyTorch: {step_count * 100} MB ({torch.cuda.memory_allocated() / (1024**2):.1f} MB)")
        _, final_used, final_free = get_smi_line()
        print(f"   nvidia-smi at OOM: Used={final_used:.1f} MiB | Free={final_free:.1f} MiB")
        print(f"   Comparison:")
        print(f"     Actual max allocatable in PyTorch : {step_count * 100} MB")
        print(f"     Initial nvidia-smi average free   : {avg_free:.1f} MiB")
        print(f"     Initial torch.cuda.mem_get_info   : {free_mb:.1f} MB")
        diff = (step_count * 100) - avg_free
        print(f"     Allocatable vs nvidia-smi free diff: {diff:+.1f} MB")

    del tensors
    torch.cuda.empty_cache()

    print("\n5. Active GPU Processes (from nvidia-smi):")
    res = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print(res.stdout.strip())

if __name__ == "__main__":
    main()
