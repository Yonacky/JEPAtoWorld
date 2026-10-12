"""Device policy: usable visible CUDA device with most free memory, else CPU.

Float64 numerical routines deliberately do not use MPS, which lacks float64.
Explicit requests fail rather than silently switching the experiment device.
"""
import os
import torch

def select_device(request=None):
    request = request or os.environ.get("JEPA_DEVICE", "auto")
    if request != "auto":
        device = torch.device(request)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("This pipeline requires CPU or CUDA (float64 support)")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA is unavailable")
            index = device.index if device.index is not None else torch.cuda.current_device()
            device = torch.device("cuda", index)
            torch.empty(1, device=device)
        return device
    candidates = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            try:
                probe = torch.empty(1, device=f"cuda:{index}")
                free, _ = torch.cuda.mem_get_info(index)
                del probe
                candidates.append((free, -index))
            except (RuntimeError, AssertionError):
                continue
    if candidates:
        return torch.device("cuda", -max(candidates)[1])
    return torch.device("cpu")

def activate_device(request=None):
    device = select_device(request)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    return device
