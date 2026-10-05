"""Screenshot the built artifact page (desktop and phone, EN and ZH) and report console errors and page overflow.
Run on della-vis1: PLAYWRIGHT_BROWSERS_PATH=.venv-shot/browsers .venv-shot/bin/python site/shot.py [out_dir]"""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

page_path = Path(__file__).resolve().parent / "_artifact/index.html"
out = Path(sys.argv[1] if len(sys.argv) > 1 else "/scratch/gpfs/GROUP/USER/project/frozen-critic/results/shots")
out.mkdir(parents=True, exist_ok=True)
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, vw, lang in (("desktop_en", 1280, "en"), ("phone_zh", 390, "zh")):
        ctx = b.new_context(viewport={"width": vw, "height": 900}, device_scale_factor=1)
        pg = ctx.new_page()
        errs = []
        pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.goto(page_path.as_uri() + f"?lang={lang}")
        pg.wait_for_timeout(2500)
        pg.evaluate("document.querySelectorAll('figure.viz').forEach(f => f.classList.add('in'))")
        over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        pg.screenshot(path=str(out / f"{name}.png"), full_page=True)
        print(name, "overflow_px", over, "errors", errs[:5])
        ctx.close()
    b.close()
