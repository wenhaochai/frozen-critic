"""Diagnostic before-train-step hook: per-parameter gradient norms at the first train steps.

Every smoke and main run shows a step-0 global grad norm of 1e11-1e12 and normal norms (1-10) from step 1 on
(not flash_qla: job 15202658; not unzeroed buffers: miles zeroes them at the start of every step). This hook wraps
optimizer.step once, so after each of the first N backward passes it records every parameter's main_grad norm and
writes runs/$RUN_NAME/grad_probe/r{rollout}_s{step}_rank{rank}.json, largest first. Use with data parallel 1, where
main_grad holds the full gradient.
  --custom-megatron-before-train-step-hook-path miles_ext.grad_probe.hook
"""
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

N_STEPS = int(os.environ.get("FC_GRAD_PROBE_STEPS", "3"))
_state = {"calls": 0, "rid": None, "sid": None}


def _dump(model):
    rows = []
    for c, chunk in enumerate(model):
        for name, p in chunk.named_parameters():
            g = getattr(p, "main_grad", None)
            if g is None:
                g = p.grad
            if g is None:
                continue
            g = g.float()
            rows.append({"name": name, "norm": g.norm().item(), "absmax": g.abs().max().item(),
                         "finite": bool(torch.isfinite(g).all()), "numel": g.numel()})
    rows.sort(key=lambda r: -r["norm"] if r["norm"] == r["norm"] else float("-inf"))
    total = sum(r["norm"] ** 2 for r in rows) ** 0.5
    rank = dist.get_rank() if dist.is_initialized() else 0
    out = Path(os.environ.get("FC_GRAD_PROBE_DIR", f"/scratch/gpfs/GROUP/USER/project/frozen-critic/runs/{os.environ.get('RUN_NAME', 'x')}/grad_probe"))
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"r{_state['rid']}_s{_state['sid']}_rank{rank}.json"
    f.write_text(json.dumps({"rollout": _state["rid"], "step": _state["sid"], "total_norm": total, "params": rows}, indent=1))
    top = ", ".join(f"{r['name']}={r['norm']:.3g}" for r in rows[:8])
    print(f"[grad_probe] rollout {_state['rid']} step {_state['sid']} rank {rank}: total {total:.4g}; top: {top}", flush=True)


def hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    _state["rid"], _state["sid"] = rollout_id, step_id
    if getattr(optimizer, "_fc_grad_probe", False):
        return
    orig = optimizer.step

    def step(*a, **k):
        if _state["calls"] < N_STEPS:
            _state["calls"] += 1
            _dump(model)
        return orig(*a, **k)

    optimizer.step = step
    optimizer._fc_grad_probe = True
