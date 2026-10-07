"""keep_latest deletes only older finished iter dirs inside runs/, never the newest, unfinished or foreign dirs."""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from miles_ext import ckpt_rotate  # noqa: E402

root = Path(__file__).resolve().parents[1] / "runs" / "_test_rotate"
shutil.rmtree(root, ignore_errors=True)
save = root / "ckpt"
for i, done in ((20, True), (40, True), (50, False), (60, True)):
    d = save / f"iter_{i:07d}"; d.mkdir(parents=True)
    if done:
        (d / "metadata.json").write_text("{}")
(save / "latest_checkpointed_iteration.txt").write_text("60")
ckpt_rotate.keep_latest(None, 2, str(save / "iter_0000060"))
left = sorted(p.name for p in save.iterdir())
print(left)
assert left == ["iter_0000050", "iter_0000060", "latest_checkpointed_iteration.txt"], left
outside = Path("/tmp") / "fc_rotate_outside" / "ckpt"
outside.mkdir(parents=True, exist_ok=True)
for i in (1, 2):
    (outside / f"iter_{i:07d}").mkdir(exist_ok=True); (outside / f"iter_{i:07d}" / "metadata.json").write_text("{}")
ckpt_rotate.keep_latest(None, 2, str(outside / "iter_0000002"))
assert (outside / "iter_0000001").exists(), "deleted outside runs/"
shutil.rmtree(root); shutil.rmtree(outside.parent)
print("OK")
