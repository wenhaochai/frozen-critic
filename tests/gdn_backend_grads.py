"""Gated DeltaNet (chunk_gated_delta_rule) gradients under fla's backend dispatch (flash_qla on Hopper) vs fla's own
Triton kernels (FLA_DISABLE_BACKEND_DISPATCH=1, read at import, so one process per mode). Same seeded inputs with the
actor's shapes (16 heads x 128, varlen packing of 1,150-token sequences, bf16); 3 calls to catch first-call garbage.
Run in images/miles-20260930.sif: python gdn_backend_grads.py OUT.pt"""
import sys

import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule

torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16
H, K, V, L, N = 16, 128, 128, 1150, 12
T = L * N
cu = torch.arange(0, T + 1, L, device=dev, dtype=torch.int32)
q = torch.randn(1, T, H, K, device=dev, dtype=dt)
k = torch.nn.functional.normalize(torch.randn(1, T, H, K, device=dev, dtype=torch.float32), dim=-1).to(dt)
v = torch.randn(1, T, H, V, device=dev, dtype=dt)
g = torch.nn.functional.logsigmoid(torch.randn(1, T, H, device=dev, dtype=torch.float32))
beta = torch.rand(1, T, H, device=dev, dtype=dt).sigmoid()
do = torch.randn(1, T, H, V, device=dev, dtype=dt)
out = {}
for call in range(3):
    xs = [t.detach().clone().requires_grad_(True) for t in (q, k, v, g, beta)]
    o, _ = chunk_gated_delta_rule(*xs, scale=K ** -0.5, cu_seqlens=cu, use_qk_l2norm_in_kernel=False)
    o.backward(do)
    out[call] = {"o": o.detach().float().cpu(), **{n: x.grad.float().cpu() for n, x in zip("qkvgb", xs)}}
    print(call, {n: (round(t.abs().max().item(), 4), bool(torch.isfinite(t).all())) for n, t in out[call].items()}, flush=True)
torch.save(out, sys.argv[1])
