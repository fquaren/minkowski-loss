"""
Resource limits on the shared compute node (node34). Enforced on import of `src`.

The node is shared: 12 CPU cores (2 sockets x 6, no SMT) and two identical RTX PRO 6000
Blackwell GPUs. This project may use **GPU 1 only** and **at most 8 cores**, so that at least
4 cores always stay free for other users. See EXPERIMENTS.md "Compute node".

    CPU   pinned to cores 4-11 (6-11 are GPU 1's NUMA node; 0-3 stay free, on GPU 0's
          socket). The affinity is inherited by DataLoader workers and process pools, so
          *everything* one job spawns shares these 8 cores, whatever its worker count.
    GPU   CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=1 (PCI 83:00.0) is set when
          nothing is set; any other explicit selection is refused. If CUDA is already
          initialised, the visible devices are checked by UUID.

`enforce()` runs from `src/__init__.py`, so any script importing from `src` is covered.
It is a no-op off node34. `MINK_NODE_LIMITS=off` disables it (do not use on node34).
"""

import os
import socket
import sys

NODE_PREFIX = "node34"
ALLOWED_CPUS = frozenset(range(4, 12))
ALLOWED_GPU_INDEX = "1"
ALLOWED_GPU_UUID = "GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4"   # PCI 00000000:83:00.0
FORBIDDEN_GPU_UUID = "GPU-a62d786a-657f-8094-ed41-3a963024e504"  # GPU 0, PCI 02:00.0


class NodeLimitError(RuntimeError):
    pass


def on_limited_node() -> bool:
    if os.environ.get("MINK_NODE_LIMITS", "").lower() == "off":
        return False
    return socket.gethostname().startswith(NODE_PREFIX)


def enforce_cpu() -> frozenset:
    """Restrict this process (and everything it later spawns) to ALLOWED_CPUS."""
    cur = frozenset(os.sched_getaffinity(0))
    new = (cur & ALLOWED_CPUS) or ALLOWED_CPUS
    if new != cur:
        os.sched_setaffinity(0, new)
    torch = sys.modules.get("torch")
    if torch is not None and torch.get_num_threads() > len(new):
        torch.set_num_threads(len(new))
    return new


def _uuid_ok(u: str) -> bool:
    u = u.lower().removeprefix("gpu-")
    return u == ALLOWED_GPU_UUID.lower().removeprefix("gpu-")


def enforce_gpu() -> str:
    """Make GPU 1 the only visible device, or fail loudly."""
    env = os.environ
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ALLOWED_GPU_INDEX
    elif cvd.strip() in ("", "-1"):
        pass  # CPU-only run: nothing visible, nothing to guard
    elif cvd.strip() == ALLOWED_GPU_INDEX:
        # Index 1 means GPU 1 only in PCI order; CUDA's default order is not nvidia-smi's.
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    elif _uuid_ok(cvd.strip()):
        pass
    else:
        raise NodeLimitError(
            f"CUDA_VISIBLE_DEVICES={cvd!r} is not allowed on this node: use GPU 1 only "
            f"(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1). GPU 0 is not ours.")

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        verify_visible_gpus()
    return env.get("CUDA_VISIBLE_DEVICES", "")


def verify_visible_gpus() -> None:
    """After CUDA init: every visible device must be GPU 1, by UUID."""
    import torch
    for i in range(torch.cuda.device_count()):
        u = str(torch.cuda.get_device_properties(i).uuid)
        if not _uuid_ok(u):
            raise NodeLimitError(
                f"visible CUDA device {i} has UUID {u}, not GPU 1 ({ALLOWED_GPU_UUID}). "
                f"CUDA was initialised before the limits could be set; set "
                f"CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 before launching.")


def enforce() -> None:
    if not on_limited_node():
        return
    enforce_cpu()
    enforce_gpu()
