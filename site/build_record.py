"""Experiment record of the frozen-critic page -> assets/data/frozen-critic-record.js (window.FC_RECORD).

Template rule (website dcfd5a1, owner 2026-10-05/07): one row per main training run, never typed; rows end with two
lines, "Main runs: X H100-hours, N jobs" and "Other compute: X H100-hours, N jobs", counted from sacct over every GPU
job of the project (GPUs x elapsed), so they sum to the sacct total. A packed job's hours are split by its task ledger
logs/packed/JOBID.tsv: each main task's GPUs x wall time to its run, everything else (smoke copies, failed tasks,
startup, gaps, idle lanes, main tasks that exited non-zero) to other compute; the job counts once, as a main job if any task is main.
Project jobs = Slurm jobs named fc-* since 2026-10-04. sacct must answer, or this fails (no guessing).
"""
import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("/scratch/gpfs/GROUP/USER/project/website/wenhaochai.github.io/assets/data/frozen-critic-record.js")
LAMBDAS = ["0.95", "0.5", "0"]
# run -> lambda. Only the character-level arm has rows (owner, 2026-10-09: the full-vocabulary runs do not make sense,
# so the page drops them); their jobs, like every other fc-* job, still count in the closing lines as other compute.
# uncentred lambda 0 was cancelled twice before start; the owner asked for it on 2026-10-11 (job 15375157)
RUNS = {**{f"ppo_char_lam{l}_s42": l for l in LAMBDAS}, **{f"ppo_charc_lam{l}_s42": l for l in LAMBDAS}}
JOB_PREFIX = {"fc-char-": "ppo_char_lam", "fc-charc-": "ppo_charc_lam"}  # job name prefix -> run name prefix


def sacct_jobs():
    out = subprocess.run(["sacct", "-u", "USER", "-S", "2026-10-04", "-X", "-n", "-P",
                          "-o", "JobID,JobName,State,ElapsedRaw,AllocTRES"], capture_output=True, text=True, check=True).stdout
    jobs = []
    for line in out.splitlines():
        jid, name, state, el, tres = line.split("|")
        if not name.startswith("fc-"):
            continue
        m = re.search(r"gres/gpu=(\d+)", tres)
        gpus = int(m.group(1)) if m else 0
        jobs.append({"id": jid, "name": name, "state": state.split()[0], "sec": int(el or 0), "gpus": gpus})
    if not jobs:
        raise SystemExit("sacct returned no fc-* jobs: refusing to write an empty record")
    return jobs


def ledger(jid):
    p = ROOT / "logs/packed" / f"{jid}.tsv"
    if not p.exists():
        return None
    return list(csv.DictReader(open(p), delimiter="\t"))


def build():
    jobs = sacct_jobs()
    run_h = {r: 0.0 for r in RUNS}  # H100-hours, for the main line
    run_wall = {r: 0.0 for r in RUNS}  # wall hours, for the row (template: rows show H100s and hours)
    run_state = {r: "planned" for r in RUNS}
    main_h = other_h = 0.0
    main_jobs = other_jobs = 0
    for j in jobs:
        total = j["gpus"] * j["sec"] / 3600
        tasks = ledger(j["id"])
        # a main task that exited non-zero is a failed task: other compute (template rule); a resumed run's later
        # successful task carries the run's hours
        mains = [t for t in (tasks or []) if t["kind"] == "main" and t["run"] in RUNS and t["exit"] == "0"]
        for t in (tasks or []):
            if t["kind"] == "main" and t["run"] in RUNS and t["exit"] != "0" and run_state[t["run"]] != "done":
                run_state[t["run"]] = "stopped"
        m = sum(int(t["gpus"]) * (int(t["end"]) - int(t["start"])) / 3600 for t in mains)
        k = min(1.0, total / m) if m else 1.0  # a ledger claiming more than the job got (duplicate line) is scaled down
        for t in mains:
            run_h[t["run"]] += k * int(t["gpus"]) * (int(t["end"]) - int(t["start"])) / 3600
            run_wall[t["run"]] += (int(t["end"]) - int(t["start"])) / 3600
            run_state[t["run"]] = "done"
        m = min(m, total)
        main_h += m
        other_h += total - m
        if total > 0:
            if mains:
                main_jobs += 1
            else:
                other_jobs += 1
    # live jobs override the ledger (a run resumed after a failed task is running again); the job name lists its
    # lambdas, fc-ppo-0.95-0.5 -> {"0.95", "0.5"} (exact tokens: "0" must not match "0.95")
    for j in jobs:
        pre = next((p for p in JOB_PREFIX if j["name"].startswith(p)), None)
        if j["state"] in ("PENDING", "RUNNING") and pre:
            lams = set(j["name"][len(pre):].split("-"))
            for r, l in RUNS.items():
                if r == f"{JOB_PREFIX[pre]}{l}_s42" and l in lams and run_state[r] != "done":
                    if j["state"] == "RUNNING" or run_state[r] != "running":
                        run_state[r] = "running" if j["state"] == "RUNNING" else "queued"
    total = sum(j["gpus"] * j["sec"] / 3600 for j in jobs)
    assert abs(main_h + other_h - total) < 1e-6, (main_h, other_h, total)
    def setup(r, l):
        c = r.startswith("ppo_charc")
        return (f"PPO with the frozen critic{', values minus the prompt value' if c else ''}, 27 output tokens, format reward 0.1, GAE λ = {l}",
                f"PPO + 冻结 critic{'，value 减去题目的 value' if c else ''}，27 个输出 token，format reward 0.1，GAE λ = {l}",
                [["3", "c-center"]] if c else [["2", "c-arith"]])
    rows = [{"run": r, "lam": l, "setup_en": setup(r, l)[0], "setup_zh": setup(r, l)[1],
             "actor": "Qwen3.5-arch 0.46B (dense, random init, 100-token vocabulary)", "gpus": 4,
             "hours": round(run_wall[r], 1) if run_wall[r] else None, "state": run_state[r], "figs": setup(r, l)[2]}
            for r, l in RUNS.items()]
    rec = {"rows": rows, "main": {"h100h": round(main_h, 1), "jobs": main_jobs},
           "other": {"h100h": round(other_h, 1), "jobs": other_jobs}, "sacct_total_h100h": round(total, 2)}
    OUT.write_text("/* generated by frozen-critic/site/build_record.py; do not edit */\nwindow.FC_RECORD = "
                   + json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + ";\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}), [(r["run"], r["state"], r["hours"]) for r in rows])
    return rec


if __name__ == "__main__":
    build()
