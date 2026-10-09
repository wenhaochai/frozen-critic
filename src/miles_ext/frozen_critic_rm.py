"""miles reward function (--custom-rm-path miles_ext.frozen_critic_rm.reward_func, --reward-key score).

Returns the strict GSM8K reward `score` (1 if the last \\boxed{} holds the gold number), the frozen critic's pre-action
values V(prompt, response[:t]) for every response token, and `train_score`, the reward PPO trains on (--reward-key):
score for a finished response; for a response cut at the length limit, V(prompt, response), the critic's value after the
whole truncated sequence (the episode did not end, so its return is the bootstrap, not the grade: adding the grade would
pay a correct answer more for never stopping). miles bootstraps the value after the last token with 0, which for a cut response
(about 99% for a random actor) would add an action-independent -V spike at the end whose reach grows with GAE lambda;
the bootstrap term removes it. Values come from src/value_server.py at $FC_VALUE_URL; the response ids are critic ids.

Format reward (owner, 2026-10-09): train_score also gets +$FC_FORMAT_REWARD (default 0) when a finished response's last
\\boxed{} holds a single number. Not for a cut response, for the reason above: it would pay the actor for never stopping.
`score` stays the strict grade.

Character-level actor ($FC_CRITIC_ID_MAP set, src/make_char_actor.py): sample.tokens are actor ids; each maps to one
critic id, and the critic reads the prompt string with its own tokenizer ($FC_CRITIC_TOKENIZER), so it sees the prompt
exactly as it did in training.

Centred values ($FC_CENTER_VALUES=1, owner 2026-10-09, the causal test of the stop penalty): every value and the
truncation bootstrap become V - V(prompt), where V(prompt) = values[0], the critic's value before the first response
token. The per-token shaping V(t+1) - V(t) is unchanged; only the stop token changes, from r - V(s_end), about 0 - 0.5,
to r - (V(s_end) - V(prompt)), about 0. The reward of a finished response is unchanged. reward['v_prompt'] keeps the
offset, so raw values are values + v_prompt. With $FC_ALLOWED_IDS set, every response id must be an allowed id: a token outside the
mask would get log-prob -inf on both sides and silently drop out of the loss (audit, 2026-10-09).
"""
import json
import os
from functools import lru_cache

import aiohttp

from common import grade, last_boxed, normalize_number

_URL = os.environ.get("FC_VALUE_URL", "")
_FORMAT = float(os.environ.get("FC_FORMAT_REWARD", "0"))
_MAP_PATH = os.environ.get("FC_CRITIC_ID_MAP") or None
_ALLOWED_PATH = os.environ.get("FC_ALLOWED_IDS") or None
_CENTER = os.environ.get("FC_CENTER_VALUES", "0") == "1"
_CRITIC_TOK = os.environ.get("FC_CRITIC_TOKENIZER", "/scratch/gpfs/GROUP/USER/project/frozen-critic/models/actor-q35d1b-lm")
_SESSIONS = {}  # one client session per event loop (a session is bound to the loop that created it)


async def _session():
    import asyncio
    loop = asyncio.get_running_loop()
    s = _SESSIONS.get(id(loop))
    if s is None or s.closed:
        s = _SESSIONS[id(loop)] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
    return s


@lru_cache(maxsize=None)
def _cmap():
    return json.load(open(_MAP_PATH))


@lru_cache(maxsize=None)
def _allowed():
    return frozenset(json.load(open(_ALLOWED_PATH)))


@lru_cache(maxsize=None)
def _critic_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(_CRITIC_TOK)


def critic_inputs(sample, response_ids):
    """(prompt ids, response ids) in the critic's vocabulary."""
    n = len(response_ids)
    if _MAP_PATH is None:  # the actor uses the critic's vocabulary
        return list(sample.tokens[: len(sample.tokens) - n]), list(response_ids)
    assert isinstance(sample.prompt, str), type(sample.prompt)
    cmap = _cmap()
    return _critic_tokenizer().encode(sample.prompt, add_special_tokens=False), [cmap[i] for i in response_ids]


async def critic_values(prompt_ids, response_ids):
    if not response_ids:
        return [], None
    s = await _session()
    async with s.post(f"{_URL}/score", json={"prompt_ids": prompt_ids, "response_ids": response_ids}) as resp:
        if resp.status != 200:
            raise RuntimeError(f"value server {resp.status}: {(await resp.text())[:500]}")
        body = await resp.json()
    return body["values"], body["final"]


async def reward_func(args, sample, **kwargs):
    assert _URL, "FC_VALUE_URL is not set"
    n_resp = sample.response_length
    response_ids = list(sample.tokens[len(sample.tokens) - n_resp:])
    if _ALLOWED_PATH is not None:
        # these two paths bypass the mask (sampling-support replay replaces it in the trainer; prefill recomputes
        # rollout log-probs without the SGLang processor)
        assert not getattr(args, "use_sampling_support_replay", False) and not getattr(args, "recompute_logprobs_via_prefill", False)
        bad = set(response_ids) - _allowed()
        assert not bad, f"response ids outside the action mask: {sorted(bad)[:10]}"
    values, final = await critic_values(*critic_inputs(sample, response_ids))
    assert len(values) == n_resp, (len(values), n_resp)
    v_prompt = values[0] if values else 0.0
    if _CENTER:
        values = [v - v_prompt for v in values]
        final = None if final is None else final - v_prompt
    score = 1.0 if grade(sample.response, sample.label) else 0.0
    status = getattr(sample.status, "value", sample.status)
    truncated = status == "truncated"
    fmt = 1.0 if normalize_number(last_boxed(sample.response)) is not None else 0.0
    train_score = final if truncated and final is not None else score + _FORMAT * fmt
    return {"score": score, "train_score": train_score, "values": values, "critic_final": final,
            "truncated": truncated, "format": fmt, "v_prompt": v_prompt, "centered": _CENTER}
