"""miles reward function (--custom-rm-path miles_ext.frozen_critic_rm.reward_func, --reward-key score).

Returns the strict GSM8K reward `score` (1 if the last \\boxed{} holds the gold number), the frozen critic's pre-action
values V(prompt, response[:t]) for every response token, and `train_score`, the reward PPO trains on (--reward-key):
score for a finished response; for a response cut at the length limit, score + V(prompt, response), the critic's value
after the whole truncated sequence. miles bootstraps the value after the last token with 0, which for a cut response
(about 99% for a random actor) would add an action-independent -V spike at the end whose reach grows with GAE lambda;
the bootstrap term removes it. Values come from src/value_server.py at $FC_VALUE_URL; the response ids are critic ids.
"""
import os

import aiohttp

from common import grade

_URL = os.environ.get("FC_VALUE_URL", "")
_SESSIONS = {}  # one client session per event loop (a session is bound to the loop that created it)


async def _session():
    import asyncio
    loop = asyncio.get_running_loop()
    s = _SESSIONS.get(id(loop))
    if s is None or s.closed:
        s = _SESSIONS[id(loop)] = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
    return s


async def critic_values(prompt_ids, response_ids):
    if not response_ids:
        return [], None
    s = await _session()
    async with s.post(f"{_URL}/score", json={"prompt_ids": prompt_ids, "response_ids": response_ids}) as resp:
        body = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"value server: {body}")
    return body["values"], body["final"]


async def reward_func(args, sample, **kwargs):
    assert _URL, "FC_VALUE_URL is not set"
    n_resp = sample.response_length
    prompt_ids = list(sample.tokens[: len(sample.tokens) - n_resp])
    response_ids = list(sample.tokens[len(sample.tokens) - n_resp:])
    values, final = await critic_values(prompt_ids, response_ids)
    assert len(values) == n_resp, (len(values), n_resp)
    score = 1.0 if grade(sample.response, sample.label) else 0.0
    status = getattr(sample.status, "value", sample.status)
    truncated = status == "truncated"
    train_score = score + (final if truncated and final is not None else 0.0)
    return {"score": score, "train_score": train_score, "values": values, "critic_final": final,
            "truncated": truncated}
