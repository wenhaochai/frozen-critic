"""miles --custom-megatron-post-save-hook-path miles_ext.ckpt_rotate.keep_latest: after a save is finalized, delete
the run's older finished checkpoints (iter_NNNNNNN dirs holding metadata.json) and keep the newest one, which is all a
resume needs (eval runs in training, so old checkpoints serve no model selection). A 1B actor with Adam is ~12-16 GB
per save; 15 saves per run would hold ~200 GB. Simplified from miles-q38-build/miles_team/ckpt_rotate.py (no critic
dir here). Deletes only inside this project's runs/ tree."""
import re
import shutil
import sys
from pathlib import Path

_ITER = re.compile(r"iter_(\d{7})")
_ROOT = Path(__file__).resolve().parents[2] / "runs"


def finished(save_dir: Path) -> list[int]:
    return sorted(int(m.group(1)) for d in save_dir.glob("iter_*")
                  if (m := _ITER.fullmatch(d.name)) and (d / "metadata.json").is_file())


def keep_latest(args, rollout_id, checkpoint_dir, hf_checkpoint_dir=None):
    cur = Path(checkpoint_dir).resolve()
    m = _ITER.fullmatch(cur.name)
    save_dir = cur.parent
    if m is None or not cur.is_dir() or _ROOT.resolve() not in save_dir.parents:
        print(f"[ckpt_rotate] skip: unexpected checkpoint dir {cur}", file=sys.stderr, flush=True)
        return
    newest = int(m.group(1))
    for i in finished(save_dir):
        if i < newest:
            shutil.rmtree(save_dir / f"iter_{i:07d}", ignore_errors=True)
            print(f"[ckpt_rotate] removed {save_dir / f'iter_{i:07d}'} (keeping {newest})", flush=True)
