import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch_npu
import random
import time
from typing import Tuple

# Try to load the vllm_ascend runtime.
try:
    from vllm_ascend.utils import enable_custom_op
    enable_custom_op()
except ImportError:
    pass

from torch.distributed.distributed_c10d import _get_default_group

def get_hcomm(rank: int):
    """Return the HCCL communicator name."""
    try:
        if torch.__version__ > "2.0.1":
            return _get_default_group()._get_backend(torch.device("npu")).get_hccl_comm_name(rank)
        else:
            return _get_default_group().get_hccl_comm_name(rank)
    except Exception as e:
        return f"mock_hcomm_{rank}"

@torch.inference_mode()
def run_dfc_kernel(m: int, k: int, inter: int, e: int, topk: int, world_size: int):
    """
    Minimal operator execution helper running in a child process.
    """
    torch_npu.npu.set_device(0)
    torch_npu.npu.config.allow_internal_format = True
    
    # 1. Initialize a single-card communication group.
    dist.init_process_group(
        backend="hccl", rank=0, world_size=1,
        init_method=f"tcp://127.0.0.1:{random.randint(20000, 30000)}"
    )
    hcomm = get_hcomm(0)
    
    # 2. Build input tensors.
    n1, n2 = 2 * inter, k
    x = torch.randn((m, k), dtype=torch.bfloat16).npu()
    expert_idx = torch.randint(0, world_size * e, (m, topk), dtype=torch.int32).npu()
    probs = torch.randn((m, topk), dtype=torch.float32).npu()
    w1_raw = torch.randint(-1, 1, (e, k, n1), dtype=torch.int8).npu()
    w2_raw = torch.randint(-1, 1, (e, inter, n2), dtype=torch.int8).npu()
    s1_raw = torch.zeros((e, n1), dtype=torch.int64).npu()
    s2_raw = torch.zeros((e, n2), dtype=torch.int64).npu()
    
    w1_list = [torch_npu.npu_format_cast(w1_raw[i], 29) for i in range(e)]
    w2_list = [torch_npu.npu_format_cast(w2_raw[i], 29) for i in range(e)]
    s1_list = [s1_raw[i] for i in range(e)]
    s2_list = [s2_raw[i] for i in range(e)]
    
    out = torch.empty((m, k), dtype=torch.bfloat16).npu()
    expert_token_nums = torch.zeros((1, e), dtype=torch.int32).npu()
    
    # 3. Execute and force synchronization. Hardware faults raise here and
    # terminate the child process.
    torch.ops._C_ascend.dispatch_ffn_combine(
        x=x, weight1=w1_list, weight2=w2_list,
        expert_idx=expert_idx, scale1=s1_list, scale2=s2_list,
        probs=probs, group=hcomm, max_output_size=65536,
        out=out, expert_token_nums=expert_token_nums
    )
    torch_npu.npu.synchronize()
    dist.destroy_process_group()

def run_dfc_worker(m, k, inter, e, topk, q):
    """
    Wrapper executed inside the child process.
    """
    try:
        run_dfc_kernel(m, k, inter, e, topk, world_size=1)
        q.put((True, "SUCCESS"))
    except Exception as ex:
        msg = str(ex).split('\n')[0]
        q.put((False, msg))

def probe_shape(m, k, inter, e, topk) -> Tuple[bool, str]:
    """
    Probe a shape in a child process so an aicore exception cannot kill
    the parent process.
    """
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    
    p = ctx.Process(target=run_dfc_worker, args=(m, k, inter, e, topk, queue))
    p.start()
    p.join(timeout=30)  # Communication probes should finish quickly; use a
    # 30s timeout to avoid hangs.
    
    if p.is_alive():
        p.terminate()
        return False, "TIMEOUT/HANG"
    
    if p.exitcode != 0:
        # A non-zero exit code usually indicates a hardware crash.
        return False, f"CRASH (ExitCode {p.exitcode})"
    
    if not queue.empty():
        return queue.get()
    return False, "UNKNOWN FAILURE"

def main():
    # Probe parameter sets.
    sweep_configs = {
        "Hidden Size (K)": [
            ("K=16", 1, 16, 256, 8, 1),
            ("K=31 (Odd)", 1, 31, 256, 8, 1),
            ("K=32 (Align32)", 1, 32, 256, 8, 1),
            ("K=48 (Align16)", 1, 48, 256, 8, 1),
            ("K=64", 1, 64, 256, 8, 1),
            ("K=127", 1, 127, 256, 8, 1),
            ("K=128", 1, 128, 256, 8, 1),
            ("K=7168", 1, 7168, 2048, 8, 1),
        ],
        "Intermediate (Inter)": [
            ("Inter=16", 1, 512, 16, 8, 1),
            ("Inter=127", 1, 512, 127, 8, 1),
            ("Inter=128", 1, 512, 128, 8, 1),
            ("Inter=511", 1, 512, 511, 8, 1),
            ("Inter=512", 1, 512, 512, 8, 1),
        ],
        "Tokens (M)": [
            ("M=1", 1, 512, 256, 8, 1),
            ("M=7", 7, 512, 256, 8, 1),
            ("M=8", 8, 512, 256, 8, 1),
        ],
        "Experts (E)": [
            ("E=1", 1, 512, 256, 1, 1),
            ("E=64", 1, 512, 256, 64, 1),
            ("E=256", 1, 512, 256, 256, 1),
            ("E=512", 1, 512, 256, 512, 1),
            ("E=1024", 1, 512, 256, 1024, 1),
        ]
    }

    results = []
    print("\nStarting Multi-Process Constraint Probing...")
    print("="*95)
    print(f"{'Category':<20} | {'Test Case':<20} | {'Result':<10} | {'Details'}")
    print("-" * 95)

    for category, cases in sweep_configs.items():
        for name, m, k, inter, e, topk in cases:
            success, msg = probe_shape(m, k, inter, e, topk)
            status = "\033[92mPASS\033[0m" if success else "\033[91mFAIL\033[0m"
            print(f"{category:<20} | {name:<20} | {status:<10} | {msg}")
            results.append((category, name, success, msg))
            # Pause briefly so the NPU driver can reclaim resources.
            time.sleep(0.5)

    print("-" * 95)
    print("\nPROBING CONCLUSION & SHAPE GENERATION GUIDELINES:")
    
    def get_alignment_requirement(cat_name):
        cat_results = [r for r in results if r[0] == cat_name]
        if not cat_results: return "Unknown"
        
        # Extract numeric values and pass/fail status.
        # Example format: ("K=31 (Odd)", 1, 31, ...) -> use numeric value 31.
        data = []
        for cat, name, succ, msg in cat_results:
            try:
                val = int(name.split('=')[1].split()[0])
                data.append((val, succ))
            except Exception: continue
        
        if all(s for v, s in data): return "No strict alignment detected (within tested range)"
        
        # Check alignment constraints.
        for align in [32, 16]:
            non_aligned_failed = all(not s for v, s in data if v % align != 0)
            aligned_passed = any(s for v, s in data if v % align == 0)
            if non_aligned_failed and aligned_passed:
                return f"STRICT {align}-ALIGNMENT REQUIRED (Violations cause CRASH)"
        
        return "Complex constraints detected (Check specific failures)"

    # 1. Hidden Size (K)
    k_req = get_alignment_requirement("Hidden Size (K)")
    print(f"  - [HIDDEN SIZE (K)]: {k_req}")
    
    # 2. Intermediate (Inter)
    i_req = get_alignment_requirement("Intermediate (Inter)")
    print(f"  - [INTERMEDIATE (Inter)]: {i_req}")

    # 3. Tokens (M)
    m_req = get_alignment_requirement("Tokens (M)")
    print(f"  - [TOKENS (M)]: {m_req}")

    # 4. Experts (E)
    e_pass = [int(r[1].split('=')[1].split()[0]) for r in results if r[0] == "Experts (E)" and r[2]]
    if e_pass:
        print(f"  - [EXPERTS (E)]: Max verified E={max(e_pass)} per card. (Ensure E is multiple of 8 for optimal performance)")
    
    print("\nACTIONABLE ADVICE FOR generate_shape_grid.py:")
    if "32-ALIGNMENT" in k_req:
        print("    * Set alignment=32 for 'hidden' parameter.")
    if "16-ALIGNMENT" in i_req or "32-ALIGNMENT" in i_req:
        align = 32 if "32" in i_req else 16
        print(f"    * Set alignment={align} for 'inter' / 'n_dim' parameter.")
    if "No strict" not in m_req and "Unknown" not in m_req:
        print("    * Caution: Token count (M) shows alignment constraints. Consider power-of-2 or 8-align.")
    
    print("="*95 + "\n")

if __name__ == "__main__":
    main()
