"""Phase 0 checks of the frozen Limite critic on one H100 (.venv-critic).

Subcommands (each writes results/<name>.json):
  repro      model-card example with eager and FlashAttention-2, plus packed-vs-single agreement on real samples
  rank       AUC of last / mean / raw-final values on graded GSM8K samples, value curves, drop positions
  probes     random-init actor samples and constructed inputs (uniform ids, small vocab, shuffled, answer-only,
             wrong final answer, empty), compared with the values of real solutions
  ascent     greedy value ascent: from an empty response, append the candidate token with the highest value
  speed      packed scoring throughput and latency
"""
import argparse
import json
import random
import time

import numpy as np
import torch

import critic
from common import CRITIC_DIR, CRITIC_EOS, QWEN25_VOCAB, RESULTS, load_gsm8k, user_prompt
from value_model import tokenize_pair

import os
RAW = RESULTS / "raw"
DEVICE = os.environ.get("FC_DEVICE", "cuda")
SMOKE = int(os.environ.get("FC_SMOKE", "0"))   # >0: keep this many records per generator, write *_smoke.json
NORMAL_GENS = ["qwen0.5b-instruct", "qwen1.5b-instruct", "qwen-math1.5b-instruct", "r1distill1.5b"]
RANDOM_GENS = ["random0.5b", "random1.5b"]
N_BINS = 20


def save(name, obj):
    path = RESULTS / f"{name}{'_smoke' if SMOKE else ''}.json"
    path.write_text(json.dumps(obj, indent=1, default=float))
    print(f"wrote {path}")


_PROMPT_CACHE = {}


def prompt_ids(question):
    if question not in _PROMPT_CACHE:
        _PROMPT_CACHE[question] = tokenize_pair(str(CRITIC_DIR), user_prompt(question), "a")[0]
    return _PROMPT_CACHE[question]


def load_gen(name):
    path = RAW / f"gen_{name}.jsonl"
    recs = [json.loads(l) for l in open(path)] if path.exists() else []
    return recs[:SMOKE] if SMOKE else recs


_TOK = []


def _critic_tok():
    if not _TOK:
        from transformers import AutoTokenizer
        _TOK.append(AutoTokenizer.from_pretrained(str(CRITIC_DIR)))
    return _TOK[0]


def auc(pos, neg):
    """P(score of a correct sample > score of an incorrect one), ties count half."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def within_problem_auc(recs, key):
    """Pairs (correct, incorrect) of the same problem; the ranking an RL group baseline needs."""
    by = {}
    for r in recs:
        by.setdefault(r["idx"], []).append(r)
    wins, n = 0.0, 0
    for rs in by.values():
        c = [r[key] for r in rs if r["correct"]]
        w = [r[key] for r in rs if not r["correct"]]
        for a in c:
            for b in w:
                wins += 1.0 if a > b else 0.5 if a == b else 0.0
                n += 1
    return (wins / n if n else None), n


def bootstrap_auc(recs, key, reps=1000, seed=0):
    """95% interval of the pooled AUC, resampling problems (samples of one problem are correlated)."""
    rng = np.random.default_rng(seed)
    by = {}
    for r in recs:
        by.setdefault(r["idx"], []).append(r)
    keys = list(by)
    vals = []
    for _ in range(reps):
        pick = rng.choice(len(keys), len(keys), replace=True)
        rs = [r for k in pick for r in by[keys[k]]]
        a = auc([r[key] for r in rs if r["correct"]], [r[key] for r in rs if not r["correct"]])
        if a is not None:
            vals.append(a)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))] if vals else None


def binned(curve, n_bins=N_BINS):
    """Mean value in n_bins equal fractions of the response."""
    edges = np.linspace(0, len(curve), n_bins + 1).round().astype(int)
    return [float(curve[a:max(b, a + 1)].mean()) if a < len(curve) else float(curve[-1])
            for a, b in zip(edges[:-1], edges[1:])]


def steepest_drop(curve, w=16):
    """Index t where V[t] - V[t-w] is most negative (the largest fall over w tokens)."""
    if len(curve) <= w:
        return None, None
    d = curve[w:] - curve[:-w]
    t = int(d.argmin())
    return t + w, float(d[t])


def score_records(model, recs, variant="raw", rows=None, max_tokens=65536):
    """Attach critic statistics to generation records."""
    q = {r["idx"]: r["question"] for r in (rows or load_gsm8k("test"))}
    items = []
    for i, r in enumerate(recs):
        ids = r["response_ids"]
        if variant == "think":  # wrap as a reasoning span: '<think>\n' + text, ids re-tokenized
            ids = _critic_tok().encode("<think>\n" + r["text"], add_special_tokens=False)
            ids = ids + ([CRITIC_EOS] if r["finished"] else [])
        if not ids:
            ids = [CRITIC_EOS]
        items.append((i, prompt_ids(q[r["idx"]]), ids))
    scores = critic.score_items(model, items, max_tokens=max_tokens)
    out = []
    for i, r in enumerate(recs):
        pre = scores[i]["pre"]
        t_drop, drop = steepest_drop(pre)
        out.append({"gen": r["gen"], "idx": r["idx"], "correct": r["correct"], "finished": r["finished"],
                    "n": len(pre), "prompt_value": float(pre[0]), "last": float(pre[-1]),
                    "mean": float(pre.mean()), "min": float(pre.min()), "max": float(pre.max()),
                    "raw_final": scores[i]["raw_last"], "bins": binned(pre),
                    "t_drop": t_drop, "drop": drop, "_pre": pre})
    return out


def summarize(recs):
    c = [r for r in recs if r["correct"]]
    w = [r for r in recs if not r["correct"]]
    s = {"n": len(recs), "n_correct": len(c), "accuracy": len(c) / max(len(recs), 1),
         "finished_frac": float(np.mean([r["finished"] for r in recs])) if recs else None,
         "mean_len": float(np.mean([r["n"] for r in recs])) if recs else None}
    for key in ("last", "mean", "raw_final", "min"):
        s[f"auc_{key}"] = auc([r[key] for r in c], [r[key] for r in w])
        s[f"auc_{key}_ci"] = bootstrap_auc(recs, key) if c and w else None
        s[f"within_auc_{key}"], s["within_pairs"] = within_problem_auc(recs, key)
        s[f"{key}_correct_mean"] = float(np.mean([r[key] for r in c])) if c else None
        s[f"{key}_incorrect_mean"] = float(np.mean([r[key] for r in w])) if w else None
    s["prompt_value_mean"] = float(np.mean([r["prompt_value"] for r in recs]))
    s["curve_correct"] = np.mean([r["bins"] for r in c], 0).tolist() if c else None
    s["curve_incorrect"] = np.mean([r["bins"] for r in w], 0).tolist() if w else None
    # Where incorrect solutions lose value: steepest fall as a fraction of the response.
    for tag, rs in (("incorrect", w), ("correct", c)):
        f = [r["t_drop"] / r["n"] for r in rs if r["t_drop"] is not None]
        s[f"drop_pos_{tag}"] = np.histogram(f, bins=10, range=(0, 1))[0].tolist() if f else None
        s[f"drop_size_{tag}_mean"] = float(np.mean([r["drop"] for r in rs if r["drop"] is not None])) if f else None
    return s


def cmd_repro(args):
    out = {}
    P = "How many positive divisors does 60 have?\n\n" + "Please reason step by step, and put your final answer within \\boxed{}."
    R = r"The prime factorisation is 60 = 2^2 * 3 * 5, so the number of divisors is (2+1)(1+1)(1+1) = 12. The answer is \boxed{12}."
    p, r = tokenize_pair(str(CRITIC_DIR), P, R)
    for backend in ("eager", "flash_attention_2"):
        m = critic.load(DEVICE, backend)
        s = m.score_tokens(p, r)
        lat = critic.timed(m.score_tokens, p, r, warmup=2, reps=5)
        out[backend] = {"prompt_value": s["prompt_value"], "mean": s["mean_response_pre_action_value"],
                        "latency_ms_100tok": lat * 1e3, "pre": s["pre_action_values"]}
        print(backend, out[backend]["prompt_value"], out[backend]["mean"], f"{lat*1e3:.1f} ms")
        del m
        torch.cuda.empty_cache()
    out["card"] = {"eager": [0.696658, 0.680541], "flash_attention_2": [0.696323, 0.680799]}
    out["max_abs_eager_vs_fa2"] = float(np.max(np.abs(np.array(out["eager"]["pre"]) - np.array(out["flash_attention_2"]["pre"]))))
    # Packed (one FA2 call over many sequences) against the released single-sequence API on real samples.
    m = critic.load(DEVICE, "flash_attention_2")
    rows = load_gsm8k("test")
    q = {r["idx"]: r["question"] for r in rows}
    recs = [x for g in NORMAL_GENS + RANDOM_GENS for x in load_gen(g)[:16]]
    items = [(i, prompt_ids(q[x["idx"]]), x["response_ids"]) for i, x in enumerate(recs)]
    out["packed_vs_single_max_abs"] = critic.verify_packing(m, items)
    out["packed_vs_single_n_seqs"] = len(items)
    out["packed_vs_single_tokens"] = sum(len(a) + len(b) for _, a, b in items)
    print("packed vs single max |diff|", out["packed_vs_single_max_abs"], "over", len(items), "sequences")
    save("repro", out)


def cmd_rank(args):
    m = critic.load(DEVICE, "flash_attention_2")
    rows = load_gsm8k("test")
    res, examples = {}, {}
    for g in NORMAL_GENS:
        recs = load_gen(g)
        if not recs:
            continue
        for variant in (["raw", "think"] if g != "r1distill1.5b" else ["raw"]):
            t0 = time.time()
            sc = score_records(m, recs, variant, rows)
            s = summarize(sc)
            s["score_seconds"] = time.time() - t0
            res[f"{g}|{variant}"] = s
            print(g, variant, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()
                               if k.startswith(("auc_last", "auc_mean", "within_auc_last", "accuracy", "n_correct"))})
            if variant == "raw":
                # Full curves of a few samples, and the text around the steepest drop of incorrect ones.
                rng = random.Random(0)
                pick = rng.sample(range(len(sc)), min(24, len(sc)))
                examples[g] = []
                for i in pick:
                    r, x = recs[i], sc[i]
                    ex = {"idx": r["idx"], "correct": r["correct"], "gold": r["gold"], "curve": x["_pre"].tolist(),
                          "text": r["text"][:6000]}
                    if x["t_drop"] is not None:
                        ids = r["response_ids"]
                        a, b = max(0, x["t_drop"] - 40), min(len(ids), x["t_drop"] + 10)
                        ex["drop_context"] = _critic_tok().decode(ids[a:x["t_drop"]]) + " ‖ " + _critic_tok().decode(ids[x["t_drop"]:b])
                        ex["t_drop"], ex["drop"] = x["t_drop"], x["drop"]
                    examples[g].append(ex)
                # Pooled distribution of last values for the histogram figure.
                res[f"{g}|raw"]["last_correct"] = [x["last"] for x in sc if x["correct"]]
                res[f"{g}|raw"]["last_incorrect"] = [x["last"] for x in sc if not x["correct"]]
                res[f"{g}|raw"]["mean_correct_vals"] = [x["mean"] for x in sc if x["correct"]]
                res[f"{g}|raw"]["mean_incorrect_vals"] = [x["mean"] for x in sc if not x["correct"]]
    save("rank", res)
    save("rank_examples", examples)


def _rand_ids(rng, n, vocab):
    return [int(rng.choice(vocab)) for _ in range(n)]


def small_vocab():
    """Digits, arithmetic, punctuation, whitespace and ~60 common GSM8K words, as single critic tokens."""
    tok = _critic_tok()
    words = ("the a of and to is in he she they has have had each per total how many much more less than "
             "dollars hours days minutes so then if he's her his it was were will be after before left "
             "first second times half twice buys sells gets gives eggs apples money cost price answer "
             "therefore step let us we need find").split()
    pieces = list("0123456789+-*/=().,$%:\n") + [" " + w for w in words] + [" " + d for d in "0123456789"] + ["\\boxed{", "}"]
    ids = sorted({i for p in pieces for i in tok.encode(p, add_special_tokens=False)[:1]})
    return ids


def cmd_probes(args):
    m = critic.load(DEVICE, "flash_attention_2")
    rows = load_gsm8k("test")
    rng = np.random.default_rng(0)
    tok = _critic_tok()
    out = {}
    # 1. Random-init actor samples (their own sampled ids).
    for g in RANDOM_GENS:
        recs = load_gen(g)
        if not recs:
            continue
        sc = score_records(m, recs, "raw", rows)
        pre = [x["_pre"] for x in sc]
        allv = np.concatenate(pre)
        deltas = np.concatenate([np.diff(p) for p in pre])
        out[g] = {"n": len(sc), "mean_len": float(np.mean([len(p) for p in pre])),
                  "finished_frac": float(np.mean([x["finished"] for x in sc])),
                  "prompt_value_mean": float(np.mean([p[0] for p in pre])),
                  "value_mean": float(allv.mean()), "value_std": float(allv.std()),
                  "value_p01": float(np.percentile(allv, 1)), "value_p99": float(np.percentile(allv, 99)),
                  "value_max": float(allv.max()), "value_min": float(allv.min()),
                  "delta_std": float(deltas.std()), "delta_abs_mean": float(np.abs(deltas).mean()),
                  "curve_mean": np.mean([x["bins"] for x in sc], 0).tolist(),
                  "curve_p10": np.percentile([x["bins"] for x in sc], 10, axis=0).tolist(),
                  "curve_p90": np.percentile([x["bins"] for x in sc], 90, axis=0).tolist(),
                  "curves_first64": [x["_pre"][:64].tolist() for x in sc[:8]],
                  "examples": [tok.decode(r["response_ids"][:80]) for r in recs[:4]]}
        print(g, {k: round(v, 3) for k, v in out[g].items() if isinstance(v, float)})
    # Reference: the same statistics on real solutions of the strongest normal generator.
    ref = load_gen("qwen-math1.5b-instruct") or load_gen("qwen1.5b-instruct")
    if ref:
        sc = score_records(m, ref[:SMOKE or 800], "raw", rows)
        allv = np.concatenate([x["_pre"] for x in sc])
        d = np.concatenate([np.diff(x["_pre"]) for x in sc])
        out["reference_real"] = {"gen": ref[0]["gen"], "value_mean": float(allv.mean()), "value_std": float(allv.std()),
                                 "value_p99": float(np.percentile(allv, 99)), "delta_std": float(d.std()),
                                 "delta_abs_mean": float(np.abs(d).mean()),
                                 "last_correct_mean": float(np.mean([x["last"] for x in sc if x["correct"]])),
                                 "last_incorrect_mean": float(np.mean([x["last"] for x in sc if not x["correct"]]))}
    # 2. Constructed inputs on 200 test problems.
    probs = rows[:3] if SMOKE else rows[:200]
    sv = small_vocab()
    golds = {r["idx"]: r for r in probs}
    ref_by_idx = {}
    for x in (ref or []):
        if x["correct"] and x["finished"] and x["idx"] not in ref_by_idx:
            ref_by_idx[x["idx"]] = x
    cases = {}

    def add(case, idx, text=None, ids=None):
        if ids is None:
            ids = tok.encode(text, add_special_tokens=False)
        cases.setdefault(case, []).append((idx, ids))

    for r in probs:
        i, gold = r["idx"], r["gold"]
        wrong = str(int(float(gold)) + rng.integers(1, 20)) if float(gold).is_integer() else gold + "1"
        add("empty (EOS only)", i, ids=[CRITIC_EOS])
        add("uniform ids, 256", i, ids=[int(t) for t in rng.integers(0, QWEN25_VOCAB, 256)])
        add("uniform ids, 1024", i, ids=[int(t) for t in rng.integers(0, QWEN25_VOCAB, 1024)])
        add("small-vocab ids, 256", i, ids=[int(t) for t in rng.choice(sv, 256)])
        add("<think> + uniform ids, 256", i, ids=tok.encode("<think>\n", add_special_tokens=False) + [int(t) for t in rng.integers(0, QWEN25_VOCAB, 256)])
        add("answer only, gold", i, f"The answer is \\boxed{{{gold}}}." )
        add("answer only, wrong", i, f"The answer is \\boxed{{{wrong}}}.")
        sol = r["solution"].split("####")[0].strip()
        sol = __import__("re").sub(r"<<[^>]*>>", "", sol)
        add("reference solution, gold answer", i, f"{sol}\nThe answer is \\boxed{{{gold}}}.")
        add("reference solution, wrong answer", i, f"{sol}\nThe answer is \\boxed{{{wrong}}}.")
        if i in ref_by_idx:
            ids = list(ref_by_idx[i]["response_ids"])
            body = ids[:-1]
            add("correct sample", i, ids=ids)
            perm = list(body)
            random.Random(i).shuffle(perm)
            add("correct sample, tokens shuffled", i, ids=perm + [CRITIC_EOS])
            add("correct sample, first half", i, ids=body[: len(body) // 2])
            add("correct sample, first half + repeated phrase", i,
                ids=body[: len(body) // 2] + tok.encode(" Let's think step by step." * 30, add_special_tokens=False))
    for case, lst in cases.items():
        items = [(k, prompt_ids(golds[i]["question"]), ids) for k, (i, ids) in enumerate(lst)]
        sc = critic.score_items(m, items)
        pre = [sc[k]["pre"] for k in range(len(lst))]
        out[f"probe|{case}"] = {"n": len(pre), "mean_len": float(np.mean([len(p) for p in pre])),
                                "last_mean": float(np.mean([p[-1] for p in pre])),
                                "mean_mean": float(np.mean([p.mean() for p in pre])),
                                "max_mean": float(np.mean([p.max() for p in pre])),
                                "raw_final_mean": float(np.mean([sc[k]["raw_last"] for k in range(len(lst))])),
                                "prompt_value_mean": float(np.mean([p[0] for p in pre])),
                                "last_values": [float(p[-1]) for p in pre],
                                "mean_values": [float(p.mean()) for p in pre]}
        print(f"{case:45s} n={len(pre):4d} last={out[f'probe|{case}']['last_mean']:.3f} mean={out[f'probe|{case}']['mean_mean']:.3f} max={out[f'probe|{case}']['max_mean']:.3f}")
    out["small_vocab_size"] = len(sv)
    save("probes", out)


@torch.no_grad()
def cmd_ascent(args):
    """Greedy value ascent from an empty response: at each step score K candidate next tokens and keep the
    one with the highest value after it. EOS is excluded (terminal value is the 0/1 reward, not V)."""
    m = critic.load(DEVICE, "flash_attention_2")
    rows = load_gsm8k("test")[: args.n_problems]
    tok = _critic_tok()
    sv = small_vocab()
    rng = np.random.default_rng(1)
    out = {"steps": args.steps, "k": args.k, "runs": []}
    for pool_name, pool in (("full vocabulary", None), ("small vocabulary", sv)):
        for r in rows:
            p = prompt_ids(r["question"])
            resp, traj = [], []
            v0 = critic.forward_packed(m, [p])[0][-1].item()
            t0 = time.time()
            for step in range(args.steps):
                if pool is None:
                    cands = rng.integers(0, QWEN25_VOCAB, args.k)
                    cands = cands[cands != CRITIC_EOS]
                else:
                    cands = np.array(pool)
                posts = critic.forward_packed(m, [p + resp + [int(c)] for c in cands])
                vals = torch.stack([x[-1] for x in posts]).float().cpu().numpy()
                best = int(vals.argmax())
                resp.append(int(cands[best]))
                traj.append(float(vals[best]))
            out["runs"].append({"pool": pool_name, "idx": r["idx"], "v0": v0, "traj": traj,
                                "text": tok.decode(resp), "seconds": time.time() - t0})
            print(pool_name, r["idx"], f"v0={v0:.3f} -> {traj[-1]:.3f} max={max(traj):.3f}", repr(tok.decode(resp)[:120]))
    save("ascent", out)


@torch.no_grad()
def cmd_speed(args):
    m = critic.load(DEVICE, "flash_attention_2")
    rows = load_gsm8k("test")
    q = {r["idx"]: r["question"] for r in rows}
    out = {"gpu": torch.cuda.get_device_name(0)}
    for gname in ("qwen1.5b-instruct", "random0.5b"):
        recs = load_gen(gname)
        if not recs:
            continue
        seqs = [(i, prompt_ids(q[x["idx"]]), x["response_ids"]) for i, x in enumerate(recs)]
        mean_len = float(np.mean([len(a) + len(b) for _, a, b in seqs]))
        res = {"mean_seq_tokens": mean_len}
        # Released API: one sequence per call.
        sub = seqs[:32]
        t = critic.timed(lambda: [m.score_tokens(list(a), list(b)) for _, a, b in sub], warmup=1, reps=2)
        res["single_api_ms_per_seq"] = t / len(sub) * 1e3
        res["single_api_tok_per_s"] = sum(len(a) + len(b) for _, a, b in sub) / t
        for pack in (8192, 16384, 32768, 65536, 131072):
            sub = seqs[: max(1, int(pack // mean_len) * 4)]
            torch.cuda.reset_peak_memory_stats()
            t = critic.timed(critic.score_items, m, sub, pack, warmup=1, reps=2)
            ntok = sum(len(a) + len(b) for _, a, b in sub)
            res[f"packed_{pack}"] = {"tok_per_s": ntok / t, "seqs_per_s": len(sub) / t,
                                     "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
            print(gname, pack, f"{ntok/t:,.0f} tok/s", f"{len(sub)/t:,.1f} seq/s",
                  f"{torch.cuda.max_memory_allocated()/2**30:.1f} GiB")
        out[gname] = res
    save("speed", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["repro", "rank", "probes", "ascent", "speed"])
    ap.add_argument("--n-problems", type=int, default=8)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--k", type=int, default=512)
    a = ap.parse_args()
    {"repro": cmd_repro, "rank": cmd_rank, "probes": cmd_probes, "ascent": cmd_ascent, "speed": cmd_speed}[a.cmd](a)
