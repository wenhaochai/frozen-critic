"""Build the frozen-critic Experiments page from results/*.json.

1. Writes assets/data/frozen-critic.js (window.FC_DATA) into the website clone.
2. Injects the cover SVG and site/charts.js into blogs/frozen-critic.html between their markers.
3. Builds the standalone artifact folder /home/USER/fc_artifact (page at index.html), like the other pages.

Usage (della-vis1): python site/build.py --state queued|running [--no-artifact]
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

import numpy as np

from cover import cover_svg

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
WEB = Path("/scratch/gpfs/GROUP/USER/project/website/wenhaochai.github.io")
PAGE = WEB / "blogs/frozen-critic.html"
DATA = WEB / "assets/data/frozen-critic.js"
ART = ROOT / "site/_artifact"

GENS = [("qwen0.5b-instruct", "Qwen2.5-0.5B-Instruct"), ("qwen1.5b-instruct", "Qwen2.5-1.5B-Instruct"),
        ("qwen-math1.5b-instruct", "Qwen2.5-Math-1.5B-Instruct"), ("r1distill1.5b", "R1-Distill-Qwen-1.5B")]
PROBES = [  # (key in probes.json, kind, en, zh)
    ("probe|correct sample", "real", "Correct sample", "答对的样本"),
    ("probe|reference solution, gold answer", "real", "Reference solution, right answer", "参考解答，答案正确"),
    ("probe|reference solution, wrong answer", "damaged", "Reference solution, wrong answer", "参考解答，答案错误"),
    ("probe|answer only, gold", "damaged", "Right answer alone", "只有正确答案"),
    ("probe|answer only, wrong", "damaged", "Wrong answer alone", "只有错误答案"),
    ("probe|correct sample, first half", "damaged", "First half of a correct sample", "答对样本的前一半"),
    ("probe|correct sample, first half + repeated phrase", "damaged", "First half, then one phrase 30 times", "前一半后接一句话重复 30 次"),
    ("probe|correct sample, tokens shuffled", "damaged", "Correct sample, tokens shuffled", "答对样本，token 打乱"),
    ("random-q4d1b", "random", "Random-init actor (dense Qwen3.8-Flash-Next, 1B)", "随机初始化的 actor（稠密 Qwen3.8-Flash-Next，1B）"),
    ("probe|uniform ids, 256", "random", "256 uniform random tokens", "256 个均匀随机 token"),
    ("probe|uniform ids, 1024", "random", "1,024 uniform random tokens", "1,024 个均匀随机 token"),
    ("probe|<think> + uniform ids, 256", "random", "<think>, then 256 random tokens", "<think> 后接 256 个随机 token"),
    ("probe|small-vocab ids, 256", "random", "256 random arithmetic tokens", "256 个随机算术常用 token"),
    ("probe|empty (EOS only)", "random", "Empty response", "空回答"),
]


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def build_data(state):
    rank, probes, ascent = load("rank"), load("probes"), load("ascent")
    speed, roll, actor = load("speed"), load("bench_rollout"), load("bench_actor")
    d = {"status": {k: state for k in ("auc", "curves", "probes", "ascent", "step")}, "gens": [], "record": []}
    d["status"].update({"acc": "planned", "monitor": "planned"})
    d["gens"] = [{"key": k, "label": lab} for k, lab in GENS]
    if rank:
        d["rank"] = {}
        for k, _ in GENS:
            r = rank.get(f"{k}|raw")
            if r:
                d["rank"][k] = {f: r.get(f) for f in ("accuracy", "auc_last", "auc_last_ci", "auc_mean", "auc_mean_ci",
                                                      "within_auc_last", "within_auc_last_ci", "within_auc_neg_len",
                                                      "within_auc_neg_len_ci", "auc_neg_len", "auc_prompt_value",
                                                      "curve_correct", "curve_incorrect", "n")}
        d["ref_acc"] = rank.get("qwen0.5b-instruct|raw", {}).get("accuracy")
    if probes:
        rows = []
        for key, kind, en, zh in PROBES:
            p = probes.get(key)
            if not p or not p.get("last_values"):
                continue
            v = np.array(p["last_values"], float)
            rows.append({"kind": kind, "label": {"en": en, "zh": zh}, "mean": float(v.mean()),
                         "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)), "n": int(len(v))})
        d["probes"] = rows
        e = probes.get("probe|empty (EOS only)")
        if e:
            d["prompt_value"] = float(np.mean(e["last_values"]))
    if ascent:
        a = {"v0": float(np.mean([r["v0"] for r in ascent["runs"]]))}
        for pool, key in (("full vocabulary", "full"), ("small vocabulary", "small")):
            runs = [r["traj"] for r in ascent["runs"] if r["pool"] == pool]
            if runs:
                a[key] = {"runs": [[round(x, 4) for x in r] for r in runs], "mean": np.mean(runs, 0).round(4).tolist()}
        if rank:
            lc = [x for k, _ in GENS for x in rank.get(f"{k}|raw", {}).get("last_correct", [])]
            if lc:
                a["band"] = [float(np.percentile(lc, 25)), float(np.percentile(lc, 75))]
        d["ascent"] = a
    if roll and speed and actor:
        steps = []
        crit = speed.get("random-q4d1b") or speed.get("qwen1.5b-instruct")
        crit_tps = max(v["tok_per_s"] for k, v in crit.items() if k.startswith("packed_"))
        for size in ("0.5b", "1b", "2b", "4b"):
            r, a = roll.get(f"actor-q4dense-{size}"), actor.get(size)
            if not r or not isinstance(a, dict) or not isinstance(a.get("fwd_bwd"), dict):
                continue
            n_seq = r["n_seqs"]
            # Tokens the critic and the actor process: the responses plus their prompts. Random-actor responses all run
            # to 1,024 tokens, so the prompt length is the mean scored length minus 1,024.
            prompt_tok = crit["mean_seq_tokens"] - 1024 if "random-q4d1b" in speed else 130
            tokens = r["tokens"] + n_seq * prompt_tok
            steps.append({"size": size.upper(), "params_B": r.get("params_B"), "rollout_s": r["seconds"], "critic_s": tokens / crit_tps,
                          "actor_s": tokens / a["fwd"]["tok_per_s"] + tokens / a["fwd_bwd"]["tok_per_s"],
                          "tokens": tokens})
        d["step"] = steps
    for k in ("auc", "curves"):
        if rank:
            d["status"][k] = "done"
    if probes:
        d["status"]["probes"] = "done"
    if ascent:
        d["status"]["ascent"] = "done"
    if d.get("step"):
        d["status"]["step"] = "done"
    for lam in ("0.95", "0.5", "0"):
        d["record"].append({"setup_en": f"PPO with the frozen critic, GAE λ = {lam}",
                            "setup_zh": f"PPO + 冻结 critic，GAE λ = {lam}", "actor": "Dense Qwen3.8-Flash-Next 1B, random init",
                            "lam": lam, "figs": [["6", "c-acc"], ["7", "c-monitor"]], "done": False})
    return d


def inject(page_text):
    s = re.sub(r"<!-- cover -->.*?<!-- /cover -->", lambda m: "<!-- cover -->" + cover_svg() + "<!-- /cover -->",
               page_text, flags=re.S)
    charts = (ROOT / "site/charts.js").read_text()
    s = re.sub(r"/\* charts \*/.*?(?=\n</script>)", lambda m: "/* charts */\n" + charts.rstrip(), s, count=1, flags=re.S)
    return s


def build_artifact():
    subprocess.run(["bash", "/scratch/gpfs/GROUP/USER/project/website/build_blog_artifact.sh"], check=True,
                   env={"OUT_DIR": str(ART), "PAGES_OVERRIDE": "blogs/frozen-critic.html", "PATH": "/usr/bin:/bin"},
                   stdout=subprocess.DEVNULL)
    s = (ART / "blogs/frozen-critic.html").read_text()
    s = s.replace('"../assets/', '"assets/').replace('data-base="../"', 'data-base="https://wenhaochai.com/"')
    s = s.replace("<title>Training a Policy from Scratch against a Frozen Critic | Experiments | Wenhao Chai</title>",
                  "<title>Frozen Critic, Random Actor</title>")
    assert '"../' not in s, "a relative link still climbs out of the page"
    (ART / "index.html").write_text(s)
    import shutil
    shutil.rmtree(ART / "blogs")
    return sorted(str(p.relative_to(ART)) for p in ART.rglob("*") if p.is_file())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="queued", choices=["queued", "running"])
    ap.add_argument("--no-artifact", action="store_true")
    args = ap.parse_args()
    data = build_data(args.state)
    DATA.write_text("/* generated by frozen-critic/site/build.py; do not edit */\nwindow.FC_DATA = "
                    + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    PAGE.write_text(inject(PAGE.read_text()))
    print("status", data["status"])
    if not args.no_artifact:
        print("\n".join(build_artifact()))
