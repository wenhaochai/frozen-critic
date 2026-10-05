"""Actor-side compute on one H100: forward+backward and forward-only token throughput of randomly initialised
dense actors, on equal-length sequences without padding. transformers 5.6.2 has no qwen4_exp, so each size of the
dense Qwen3.8-Flash-Next actor is measured on its qwen3_5 proxy (models/proxy-q35dense-*: same width, depth, MLP and
heads, Gated DeltaNet via flash-linear-attention; no hyper-connections, indexer or n-gram embedding). Gradient
checkpointing on every size. The optimizer step is left out (it is per parameter, not per token)."""
import argparse
import json
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from common import RESULTS

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="0.5b,1b,2b,4b")
ap.add_argument("--seq", type=int, default=1152)       # ~130 prompt + 1024 response tokens
ap.add_argument("--tokens-per-step", type=int, default=16384)
args = ap.parse_args()

try:
    import flash_attn  # noqa: F401
    attn = "flash_attention_2"
except ImportError:
    attn = "sdpa"
out = {"attn": attn, "gpu": torch.cuda.get_device_name(0)}
H100_BF16 = 989e12
for size in args.sizes.split(","):
    from common import ROOT
    cfg = AutoConfig.from_pretrained(str(ROOT / "models" / f"proxy-q35dense-{size}"))
    cfg._attn_implementation = attn
    model = AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()  # every size, so the sizes compare and 1.5B fits beside fp32 logits
    n_params = sum(p.numel() for p in model.parameters())
    n_embed = model.model.embed_tokens.weight.numel()
    bs = max(1, args.tokens_per_step // args.seq)
    ids = torch.randint(0, 151667, (bs, args.seq), device="cuda")

    def fwd_bwd():
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def fwd():
        model(input_ids=ids, logits_to_keep=0)

    res = {"params_B": n_params / 1e9, "non_embedding_params_B": (n_params - n_embed) / 1e9, "batch": bs, "seq": args.seq,
           "grad_ckpt": True}
    for name, fn in (("fwd_bwd", fwd_bwd), ("fwd", fwd)):
        try:
            torch.cuda.reset_peak_memory_stats()
            fn(); torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 3
            tps = bs * args.seq / dt
            flops_per_tok = (6 if name == "fwd_bwd" else 2) * n_params
            res[name] = {"tok_per_s": tps, "mfu_6N": tps * flops_per_tok / H100_BF16,
                         "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
        except torch.cuda.OutOfMemoryError:
            res[name] = "OOM"
            torch.cuda.empty_cache()
    out[size] = res
    print(size, json.dumps(res), flush=True)
    (RESULTS / "bench_actor.json").write_text(json.dumps(out, indent=1))
    del model
    torch.cuda.empty_cache()
(RESULTS / "bench_actor.json").write_text(json.dumps(out, indent=1))
