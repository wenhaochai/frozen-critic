"""Packed (batched) scoring for the frozen Limite value model.

The released API scores one unpadded sequence per call. Every GMPT op except attention is per token, and
both attention paths already take sequence boundaries (cu_seqlens for FlashAttention, seq_ids for eager),
so several sequences can be packed into one row with positions restarting at 0. The only blocker is
`get_cu_seqlens_from_position_ids`, which rejects position resets; `enable_packing()` swaps it for a version
that derives the boundaries from the resets. `verify_packing` checks packed against single-sequence output.
"""
import sys
import time

import torch
from torch.nn import functional as F

from common import CRITIC_DIR, CRITIC_VOCAB

sys.path.insert(0, str(CRITIC_DIR))
import modeling_backbone  # noqa: E402
from value_model import load_value_model  # noqa: E402

_ORIG_CU = modeling_backbone.get_cu_seqlens_from_position_ids


def _packed_cu_seqlens(position_ids):
    """Boundaries from position resets; each segment must be 0..len-1."""
    if position_ids.shape[0] != 1:
        raise ValueError("packed input must be one row")
    pos = position_ids[0]
    n = pos.numel()
    starts = torch.nonzero(pos == 0, as_tuple=False).flatten()
    if starts.numel() == 0 or starts[0].item() != 0:
        raise ValueError("packed positions must start at 0")
    ends = torch.cat([starts[1:], torch.tensor([n], device=pos.device)])
    lengths = ends - starts
    expected = torch.arange(n, device=pos.device) - torch.repeat_interleave(starts, lengths)
    if not torch.equal(pos.long(), expected):
        raise ValueError("positions must restart at 0 and be contiguous within each packed sequence")
    cu = torch.cat([starts, torch.tensor([n], device=pos.device)]).to(torch.int32)
    return cu, int(lengths.max().item())


def enable_packing():
    modeling_backbone.get_cu_seqlens_from_position_ids = _packed_cu_seqlens


BACKEND = None


def load(device="cuda", backend="flash_attention_2"):
    """FlashAttention-2 when installed; the eager reference backend otherwise (then packs stay <= 4096 tokens)."""
    global BACKEND
    if backend.startswith("flash_attention") and torch.device(device).type != "cuda":
        backend = "eager"
    if backend == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            print("flash_attn not installed: falling back to the eager backend")
            backend = "eager"
    model = load_value_model(str(CRITIC_DIR), device=device, attention_backend=backend)
    enable_packing()
    BACKEND = backend
    return model


@torch.no_grad()
def forward_packed(model, seqs):
    """Post-token values for each full sequence in `seqs` (lists of ids), from one packed forward."""
    lengths = [len(s) for s in seqs]
    if BACKEND == "eager" and max(lengths) > 4096:
        raise ValueError("the eager backend scores at most 4,096 tokens per sequence")
    flat = [t for s in seqs for t in s]
    if min(flat) < 0 or max(flat) >= CRITIC_VOCAB:
        raise ValueError("critic ids must lie in [0, 151667)")
    dev = model.device
    ids = torch.tensor([flat], dtype=torch.long, device=dev)
    pos = torch.cat([torch.arange(n, device=dev) for n in lengths])[None, :]
    with torch.autocast(device_type=dev.type, enabled=False):
        h = model.backbone(input_ids=ids, position_ids=pos).last_hidden_state
        post = F.linear(h.float(), model.value_head.weight, model.value_head.bias).squeeze(-1)[0]
    if not torch.isfinite(post).all():
        raise RuntimeError("non-finite critic output")
    return list(torch.split(post, lengths))


def pre_action(post, p, length):
    """V(prompt, response[:t]) for t = 0..L-1: post[P-1 : P+L-1] (the model card's alignment)."""
    return post[p - 1:p + length - 1]


def make_packs(items, max_tokens):
    """Greedy packing of (key, prompt_ids, response_ids) in input order, at most max_tokens per pack."""
    packs, cur, cur_tok = [], [], 0
    for it in items:
        n = len(it[1]) + len(it[2])
        if cur and cur_tok + n > max_tokens:
            packs.append(cur)
            cur, cur_tok = [], 0
        cur.append(it)
        cur_tok += n
    if cur:
        packs.append(cur)
    return packs


@torch.no_grad()
def score_items(model, items, max_tokens=65536, return_post_last=True):
    """items: list of (key, prompt_ids, response_ids). Returns {key: dict(pre=np array, raw_last=float)}."""
    out = {}
    if BACKEND == "eager":
        max_tokens = min(max_tokens, 4096)
    # Sort by length so packs are similar in size; results are keyed, order does not matter.
    items = sorted(items, key=lambda it: len(it[1]) + len(it[2]))
    for pack in make_packs(items, max_tokens):
        posts = forward_packed(model, [list(p) + list(r) for _, p, r in pack])
        for (key, p, r), post in zip(pack, posts):
            pre = pre_action(post, len(p), len(r)).cpu().numpy()
            out[key] = {"pre": pre, "raw_last": float(post[-1].item())}
    return out


@torch.no_grad()
def verify_packing(model, items, max_tokens=65536):
    """Max |packed - single| over all pre-action values (single = the released score_tokens API)."""
    packed = score_items(model, items, max_tokens)
    worst = 0.0
    for key, p, r in items:
        single = torch.tensor(model.score_tokens(list(p), list(r))["pre_action_values"])
        worst = max(worst, (single - torch.tensor(packed[key]["pre"])).abs().max().item())
    return worst


def timed(fn, *a, warmup=1, reps=3, **k):
    for _ in range(warmup):
        fn(*a, **k)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn(*a, **k)
    torch.cuda.synchronize()
    return (time.time() - t0) / reps
