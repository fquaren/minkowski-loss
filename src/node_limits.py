"""
Resource limits on the shared compute node. Enforced on import of `src`.

Our group is assigned one GPU, identified by its UUID (ALLOWED_GPU_UUID), not by a node:
the card has moved between nodes (node34 until 2026-11-03, then node32). The limits apply on
whichever node holds that card. That node is shared: 12 CPU cores (2 sockets x 6, no SMT) and
two identical RTX PRO 6000 Blackwell GPUs. This project may use **our GPU only** and **at most
8 cores**, so that at least 4 cores always stay free for other users. See EXPERIMENTS.md
"Compute node".

    CPU   pinned to cores 4-11 (6-11 are our GPU's NUMA node on node32/node34; 0-3 stay
          free). The affinity is inherited by DataLoader workers and process pools, so
          *everything* one job spawns shares these 8 cores, whatever its worker count.
    GPU   when nothing is set: CUDA_DEVICE_ORDER=PCI_BUS_ID + CUDA_VISIBLE_DEVICES=<our card's
          index in PCI order> (1 on node32). An explicit selection is accepted only if every
          entry is our card, by PCI-order index or by UUID. If CUDA is already initialised,
          the visible devices are checked by UUID.

The node's cards are read from /proc/driver/nvidia/gpus/*/information, which does not
initialise CUDA. `enforce()` runs from `src/__init__.py`, so any script importing from `src`
is covered. It is a no-op on a host without our card. `MINK_NODE_LIMITS=off` disables it
(do not use where our card is).
"""

import glob
import os
import sys

ALLOWED_CPUS = frozenset(range(4, 12))
ALLOWED_GPU_UUID = "GPU-9e0aab27-00b0-8d7e-df63-922075bf41b4"   # ours; PCI 83:00.0 on node32/34
FORBIDDEN_GPU_UUID = "GPU-a62d786a-657f-8094-ed41-3a963024e504"  # the other group's, beside ours


class NodeLimitError(RuntimeError):
    pass


def node_gpu_uuids() -> list:
    """UUIDs of this host's NVIDIA GPUs in PCI bus order (= CUDA_DEVICE_ORDER=PCI_BUS_ID)."""
    uuids = []
    for d in sorted(glob.glob("/proc/driver/nvidia/gpus/*")):  # dir names are PCI bus ids
        try:
            with open(os.path.join(d, "information")) as f:
                for line in f:
                    if line.startswith("GPU UUID:"):
                        uuids.append(line.split(":", 1)[1].strip())
        except OSError:
            pass
    return uuids


def on_limited_node() -> bool:
    if os.environ.get("MINK_NODE_LIMITS", "").lower() == "off":
        return False
    return any(_uuid_ok(u) for u in node_gpu_uuids())


def enforce_cpu() -> frozenset:
    """Restrict this process (and everything it later spawns) to ALLOWED_CPUS."""
    if not ALLOWED_CPUS <= set(range(os.cpu_count() or 0)):
        raise NodeLimitError(
            f"this node has {os.cpu_count()} cores, so cores {sorted(ALLOWED_CPUS)} do not fit: "
            f"our GPU has moved to a node with a different layout; update ALLOWED_CPUS.")
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
    """Make our GPU the only visible device, or fail loudly."""
    env = os.environ
    uuids = node_gpu_uuids()
    ours = next(i for i, u in enumerate(uuids) if _uuid_ok(u))
    cvd = env.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(ours)
    elif cvd.strip() in ("", "-1"):
        pass  # CPU-only run: nothing visible, nothing to guard
    else:
        entries = [e.strip() for e in cvd.split(",")]
        # An index means a card in PCI order only with CUDA_DEVICE_ORDER=PCI_BUS_ID; CUDA's
        # default order is not nvidia-smi's, so the order is set whenever an index is used.
        if all(e == str(ours) or _uuid_ok(e) for e in entries):
            if any(e.isdigit() for e in entries):
                env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        else:
            raise NodeLimitError(
                f"CUDA_VISIBLE_DEVICES={cvd!r} is not allowed on this node: use our GPU only "
                f"(CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES={ours}, or its UUID "
                f"{ALLOWED_GPU_UUID}). The other GPU is not ours.")

    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        verify_visible_gpus()
    return env.get("CUDA_VISIBLE_DEVICES", "")


def verify_visible_gpus() -> None:
    """After CUDA init: every visible device must be our GPU, by UUID."""
    import torch
    for i in range(torch.cuda.device_count()):
        u = str(torch.cuda.get_device_properties(i).uuid)
        if not _uuid_ok(u):
            raise NodeLimitError(
                f"visible CUDA device {i} has UUID {u}, not our GPU ({ALLOWED_GPU_UUID}). "
                f"CUDA was initialised before the limits could be set; set "
                f"CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES to our card before launching.")


def enforce() -> None:
    if not on_limited_node():
        return
    enforce_cpu()
    enforce_gpu()
