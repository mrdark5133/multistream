"""
Phase 0 Fix-up Task 1: Condition (b) Measurement
- 5 consecutive nvidia-smi readings (2s apart)
- 5-sample average used directly in gap analysis (no 6th query)
- Raw nvidia-smi process list
- Direct measurement: allocate torch tensors in 100 MB steps until failure
- Report allocatable MB vs nvidia-smi free MB
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

def get_raw_nvidia_smi():
    res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return res.stdout

def main():
    print("=" * 70)
    print("CONDITION (B) MEASUREMENT & GAP ANALYSIS")
    print("=" * 70)

    print("\n1. 5 Consecutive nvidia-smi Readings (2s interval):")
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

        print("\n3. Gap Analysis (5-sample average vs PyTorch query):")
        print(f"   nvidia-smi 5-sample avg: Total={avg_tot:.1f} MiB, Used={avg_used:.1f} MiB, Free={avg_free:.1f} MiB")
        print(f"   torch.cuda.mem_get_info : Free={free_mb:.1f} MB, Total={total_mb:.1f} MB")
        gap = free_mb - avg_free
        print(f"   Discrepancy (PyTorch free - nvidia-smi free): {gap:+.1f} MB")

        print("\n4. Tensor Allocation Measurement (100 MB steps until failure):")
        tensors = []
        step_mb = 100
        step_bytes = step_mb * 1024 * 1024
        elements_per_step = step_bytes // 4 # float32
        allocatable_mb = 0

        try:
            while True:
                # Allocate 100 MB tensor on cuda:0
                t = torch.empty(elements_per_step, dtype=torch.float32, device="cuda:0")
                tensors.append(t)
                allocatable_mb += step_mb
                if allocatable_mb % 500 == 0:
                    tot_cur, used_cur, free_cur = get_smi_line()
                    print(f"   Allocated {allocatable_mb} MB | nvidia-smi: Used={used_cur:.1f} MiB, Free={free_cur:.1f} MiB")
        except torch.cuda.OutOfMemoryError as e:
            print(f"   Allocation halted due to OutOfMemoryError at step {allocatable_mb + step_mb} MB.")
        except Exception as e:
            print(f"   Allocation stopped with exception: {e}")

        print(f"\n   Total Successfully Allocatable: {allocatable_mb} MB")
        print(f"   Comparison: Allocatable = {allocatable_mb} MB vs nvidia-smi 5-sample Free = {avg_free:.1f} MiB")
        if allocatable_mb > avg_free:
            print(f"   Result: PyTorch successfully allocated {allocatable_mb - avg_free:.1f} MB MORE than physical nvidia-smi free VRAM.")
            print("   Explanation: Under Windows WDDM, the OS kernel virtualizes and overcommits VRAM, paging excess buffers to host RAM/pagefile rather than failing at the physical VRAM boundary.")
        else:
            print(f"   Result: PyTorch allocated {allocatable_mb} MB before OOM.")

        # Clean up tensors
        del tensors
        torch.cuda.empty_cache()
    else:
        print("   CUDA not available in PyTorch.")

    print("\n5. Raw nvidia-smi Output and Process List:")
    print(get_raw_nvidia_smi())

if __name__ == "__main__":
    main()
