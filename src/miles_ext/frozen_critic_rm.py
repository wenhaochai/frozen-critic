"""miles reward function (--custom-rm-path miles_ext.frozen_critic_rm.reward_func, --reward-key score).

Returns the strict GSM8K reward (1 if the last \\boxed{} holds the gold number) and the frozen critic's pre-action
values V(prompt, response[:t]) for every response token, fetched from src/value_server.py at $FC_VALUE_URL. The
response ids are the actor's sampled ids, which are critic ids (the actor uses the critic's vocabulary).
"""
import os

import aiohttp

from common import grade

_URL = os.environ.get("FC_VALUE_URL", "")
_SESSION = None


async def _session():
    global _SESSION
    if _SESSION is None or _SESSION.closed:
        _SESSION = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600))
    return _SESSION


async def critic_values(prompt_ids, response_ids):
    if not response_ids:
        return []
    s = await _session()
    async with s.post(f"{_URL}/score", json={"prompt_ids": prompt_ids, "response_ids": response_ids}) as resp:
        body = await resp.json()
        if resp.status != 200:
            raise RuntimeError(f"value server: {body}")
    return body["values"]


async def reward_func(args, sample, **kwargs):
    assert _URL, "FC_VALUE_URL is not set"
    n_resp = sample.response_length
    prompt_ids = list(sample.tokens[: len(sample.tokens) - n_resp])
    response_ids = list(sample.tokens[len(sample.tokens) - n_resp:])
    values = await critic_values(prompt_ids, response_ids)
    assert len(values) == n_resp, (len(values), n_resp)
    score = 1.0 if grade(sample.response, sample.label) else 0.0
    return {"score": score, "values": values, "critic_last": values[-1] if values else None}
