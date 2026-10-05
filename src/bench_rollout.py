"""Rollout throughput of a random actor on one H100 (vLLM, random dummy weights, no EOS so every response runs to
max length, as a random policy's would). One RL-sized batch per size: prompts x samples x max_tokens."""
import argparse
import json
import time

from vllm import LLM, SamplingParams

from common import RESULTS, load_gsm8k, user_prompt

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="0.5B,1.5B,3B,7B")
ap.add_argument("--n-prompts", type=int, default=256)
ap.add_argument("--n-samples", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=1024)
args = ap.parse_args()

rows = load_gsm8k("test")[: args.n_prompts]
out = {}
for size in args.sizes.split(","):
    name = f"Qwen/Qwen2.5-{size}-Instruct"
    llm = LLM(model=name, load_format="dummy", dtype="bfloat16", seed=0, gpu_memory_utilization=0.85,
              max_model_len=args.max_tokens + 512, enable_prefix_caching=True)
    tok = llm.get_tokenizer()
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_prompt(r["question"])}], tokenize=False,
                                       add_generation_prompt=True) for r in rows]
    sp = SamplingParams(n=args.n_samples, temperature=1.0, max_tokens=args.max_tokens, ignore_eos=True, seed=0)
    llm.generate(prompts[:8], SamplingParams(max_tokens=16))  # warmup
    t0 = time.time()
    res = llm.generate(prompts, sp)
    dt = time.time() - t0
    ntok = sum(len(c.token_ids) for o in res for c in o.outputs)
    out[size] = {"seconds": dt, "tokens": ntok, "tok_per_s": ntok / dt, "n_seqs": len(prompts) * args.n_samples,
                 "max_tokens": args.max_tokens}
    print(size, json.dumps(out[size]), flush=True)
    (RESULTS / "bench_rollout.json").write_text(json.dumps(out, indent=1))
    del llm
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
(RESULTS / "bench_rollout.json").write_text(json.dumps(out, indent=1))
