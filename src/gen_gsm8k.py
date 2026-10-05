"""Sample GSM8K test solutions with vLLM and save them with critic-ready token ids.

Run in .venv-gen on one GPU. For each generator: chat-formatted prompt with the critic's instruction,
temperature 1 sampling, graded with the strict boxed-answer reward. Responses are stored as text plus
critic token ids (re-tokenized with the critic tokenizer; <|endoftext|> appended when the generation
finished). A random-init generator keeps its own sampled ids, since re-tokenizing gibberish changes them.
"""
import argparse
import json
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from common import (CRITIC_DIR, CRITIC_EOS, QWEN25_VOCAB, RESULTS, grade, load_gsm8k, user_prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-problems", type=int, default=500)
    ap.add_argument("--n-samples", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--random-init", action="store_true", help="keep sampled ids; tokenizer from --tokenizer")
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = load_gsm8k("test")[: args.n_problems]
    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    critic_tok = AutoTokenizer.from_pretrained(str(CRITIC_DIR))
    prompts = [tok.apply_chat_template([{"role": "user", "content": user_prompt(r["question"])}],
                                       tokenize=False, add_generation_prompt=True) for r in rows]
    prefill_think = prompts[0].endswith("<think>\n")  # R1-distill templates open the think span themselves

    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=0.85, max_model_len=args.max_tokens + 512, enable_prefix_caching=True)
    sp = SamplingParams(n=args.n_samples, temperature=args.temperature, top_p=1.0, max_tokens=args.max_tokens,
                        seed=args.seed)
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    gen_s = time.time() - t0

    out_path = RESULTS / "raw" / f"gen_{args.name}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_tok, n_correct, n_total = 0, 0, 0
    with open(out_path, "w") as f:
        for r, o in zip(rows, outs):
            for k, c in enumerate(o.outputs):
                finished = c.finish_reason == "stop"
                text = ("<think>\n" if prefill_think else "") + c.text
                if args.random_init:
                    resp_ids = [int(t) for t in c.token_ids]
                    assert max(resp_ids) < QWEN25_VOCAB, "random actor emitted a padding id"
                    # vLLM includes the stop token in token_ids when it stops on EOS; map any stop id to critic EOS.
                    if finished and resp_ids and resp_ids[-1] in (151643, 151645):
                        resp_ids = resp_ids[:-1]
                else:
                    resp_ids = critic_tok.encode(text, add_special_tokens=False)
                if finished:
                    resp_ids = resp_ids + [CRITIC_EOS]
                correct = grade(text, r["gold"])
                n_tok += len(c.token_ids)
                n_correct += int(correct)
                n_total += 1
                f.write(json.dumps({"gen": args.name, "idx": r["idx"], "sample": k, "gold": r["gold"],
                                    "text": text, "finished": finished, "correct": correct,
                                    "n_gen_tokens": len(c.token_ids), "response_ids": resp_ids}) + "\n")
    summary = {"gen": args.name, "model": args.model, "n": n_total, "accuracy": n_correct / n_total,
               "gen_tokens": n_tok, "gen_seconds": gen_s, "gen_tok_per_s": n_tok / gen_s,
               "max_tokens": args.max_tokens, "temperature": args.temperature}
    print(json.dumps(summary))
    with open(RESULTS / "raw" / f"gen_{args.name}.summary.json", "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
