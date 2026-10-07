"""HTTP value service for the frozen Limite critic (runs on its own GPU, outside the miles Ray pool).

POST /score {"prompt_ids": [...], "response_ids": [...]} -> {"values": [V(prompt, response[:t]) for t < L]}
Concurrent requests are gathered for up to --window-ms or --max-tokens and scored in one packed FlashAttention-2
forward (src/critic.py), so per-sample calls from the rollout workers batch themselves.
GET /health -> {"ok": true, "scored": n}
"""
import argparse
import asyncio
import time

from aiohttp import web

import critic
from common import CRITIC_VOCAB

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--window-ms", type=float, default=30.0)
ap.add_argument("--max-tokens", type=int, default=131072)
ap.add_argument("--device", default="cuda", help="cpu only for tests (eager backend)")
args = ap.parse_args()

model = critic.load(args.device, "flash_attention_2")
assert args.device != "cuda" or critic.BACKEND == "flash_attention_2"
queue: asyncio.Queue = asyncio.Queue()
stats = {"scored": 0, "tokens": 0, "batches": 0, "busy_s": 0.0}


def _score(batch):
    items = [(i, p, r) for i, (p, r, _) in enumerate(batch)]
    out = critic.score_items(model, items, max_tokens=args.max_tokens)
    return [out[i]["pre"].tolist() for i in range(len(batch))]


async def batcher():
    loop = asyncio.get_running_loop()
    while True:
        first = await queue.get()
        batch, ntok = [first], len(first[0]) + len(first[1])
        deadline = loop.time() + args.window_ms / 1000
        while ntok < args.max_tokens:
            timeout = deadline - loop.time()
            if timeout <= 0:
                break
            try:
                item = await asyncio.wait_for(queue.get(), timeout)
            except asyncio.TimeoutError:
                break
            batch.append(item)
            ntok += len(item[0]) + len(item[1])
        t0 = time.time()
        try:
            values = await loop.run_in_executor(None, _score, batch)
            for (_, _, fut), v in zip(batch, values):
                fut.set_result(v)
        except Exception as e:  # fail every waiter loudly; the reward function raises
            for _, _, fut in batch:
                if not fut.done():
                    fut.set_exception(e)
        stats["busy_s"] += time.time() - t0
        stats["scored"] += len(batch)
        stats["tokens"] += ntok
        stats["batches"] += 1


async def score(request):
    body = await request.json()
    p, r = body["prompt_ids"], body["response_ids"]
    if not p or not r:
        return web.json_response({"error": "empty prompt or response"}, status=400)
    if max(p + r) >= CRITIC_VOCAB or min(p + r) < 0:
        return web.json_response({"error": "token id outside the critic vocabulary"}, status=400)
    fut = asyncio.get_running_loop().create_future()
    await queue.put((p, r, fut))
    return web.json_response({"values": await fut})


async def health(_):
    return web.json_response({"ok": True, **stats})


async def main():
    asyncio.create_task(batcher())
    app = web.Application(client_max_size=64 * 2**20)
    app.add_routes([web.post("/score", score), web.get("/health", health)])
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, args.host, args.port).start()
    print(f"value server on {args.host}:{args.port}", flush=True)
    while True:
        await asyncio.sleep(3600)


asyncio.run(main())
