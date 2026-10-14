"""Build the frozen-critic Experiments page from results/*.json.

1. Writes assets/data/frozen-critic.js (window.FC_DATA) into the website clone.
2. Injects the cover SVG and site/charts.js into blogs/frozen-critic.html between their markers.
3. Builds the standalone artifact folder site/_artifact (page at index.html), like the other pages.
4. Sets the ?v= cache busters of the two data files to a hash of their content.

Usage (della-vis1): python site/build.py [--no-artifact]; figure states come from the record (site/build_record.py)
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


from cover import cover_svg

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
WEB = Path("/scratch/gpfs/GROUP/USER/project/website/wenhaochai.github.io")
PAGE = WEB / "blogs/frozen-critic.html"
DATA = WEB / "assets/data/frozen-critic.js"
ART = ROOT / "site/_artifact"

GENS = [("qwen0.5b-instruct", "Qwen2.5-0.5B-Instruct"), ("qwen1.5b-instruct", "Qwen2.5-1.5B-Instruct"),
        ("qwen-math1.5b-instruct", "Qwen2.5-Math-1.5B-Instruct"), ("r1distill1.5b", "R1-Distill-Qwen-1.5B")]
def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def build_data(rec):
    """Figure 1 (critic AUC, results/rank.json), figure 2 (character-level actor, phase1_char_curves.json), figure 3 (centred values).
    A panel without data shows its run's state from the record: status[fig][lambda]."""
    rank = load("rank")
    st = {r["run"]: r["state"] for r in rec["rows"]}
    d = {"status": {"auc": "queued", "arith": {l: st.get(f"ppo_char_lam{l}_s42", "planned") for l in ("0.95", "0.5", "0")},
                    "center": {l: st.get(f"ppo_charc_lam{l}_s42", "planned") for l in ("0.95", "0.5", "0")}},
         "gens": [{"key": k, "label": lab} for k, lab in GENS]}
    if rank:
        d["rank"] = {}
        for k, _ in GENS:
            r = rank.get(f"{k}|raw")
            if r:
                d["rank"][k] = {f: r.get(f) for f in ("accuracy", "auc_last", "auc_last_ci", "auc_mean", "auc_mean_ci",
                                                      "within_auc_last", "within_auc_last_ci", "within_auc_neg_len",
                                                      "within_auc_neg_len_ci")}
        d["ref_acc"] = rank.get("qwen0.5b-instruct|raw", {}).get("accuracy")
        d["status"]["auc"] = "done"
    # site/phase1_curves.py: per lambda, train [[rollout, value_last, frac_correct, frac_boxed_number]], eval [[rollout, acc]]
    p1 = load("phase1_char_curves")
    if p1:
        d["phase1"] = {l: {"train": [[x[0]] + [round(v, 4) for v in x[1:]] for x in c["train"]], "eval": c["eval"]} for l, c in p1.items()}
    p1c = load("phase1_charc_curves")  # figure 3: the same arm with centred values V - V(prompt)
    if p1c:
        d["phase1c"] = {l: {"train": [[x[0]] + [round(v, 4) for v in x[1:]] for x in c["train"]], "eval": c["eval"]} for l, c in p1c.items()}
    # figure 4: does the critic pick the answer? (src/diag_answer_pick.py, diag_answer_open.py, diag_answer_open_extra.py)
    pick, opn, ext = load("answer_pick"), load("answer_open"), load("answer_open_extra")
    if pick and opn and ext:
        r4 = lambda v: [round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)
        d["answer"] = {"pick": {"n": pick["char"]["n_problems"], "top1": r4(pick["char"]["top1"]), "chance": r4(pick["char"]["chance_top1"])},
                       "open": {"n": opn["n_problems"], "own": r4(opn["true"]["top1"]), "other": r4(opn["swapped"]["top1"]),
                                "chance": opn["chance_top1"], "const": ext["const"], "const_200": r4(ext["const_200"])}}
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
    ap.add_argument("--no-artifact", action="store_true")
    args = ap.parse_args()
    import build_record
    data = build_data(build_record.build())  # sacct-backed record (fails loudly)
    DATA.write_text("/* generated by frozen-critic/site/build.py; do not edit */\nwindow.FC_DATA = "
                    + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + ";\n")
    page = inject(PAGE.read_text())
    for f in (DATA, build_record.OUT):  # cache busters follow the content, so a browser never keeps an old record
        v = hashlib.sha1(f.read_bytes()).hexdigest()[:8]
        page = re.sub(rf"(assets/data/{re.escape(f.name)})\?v=\w+", rf"\1?v={v}", page)
    PAGE.write_text(page)
    print("status", data["status"])
    if not args.no_artifact:
        print("\n".join(build_artifact()))
