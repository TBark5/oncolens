"""Start the Streamlit app, drive it with a headless browser, and save README screenshots.

Dev-only tool (needs `pip install -r requirements-dev.txt`). It uses the locally installed
Google Chrome through Playwright, so no browser download is required. Pass
``--browser msedge`` to use Microsoft Edge instead.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "screenshots"
PORT = 8765


@contextmanager
def streamlit_server(port: int = PORT) -> Iterator[str]:
    """Run ``streamlit run app.py`` in the background for the duration of the block."""
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"), "--server.headless", "true",
         "--server.port", str(port)],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    url = f"http://localhost:{port}"
    try:
        for _ in range(60):
            try:
                if urllib.request.urlopen(f"{url}/_stcore/health", timeout=2).read() == b"ok":
                    break
            except OSError:
                time.sleep(1)
        else:
            raise RuntimeError("Streamlit did not become healthy within 60 s")
        yield url
    finally:
        proc.terminate()
        proc.wait(timeout=20)


def wait_for_app(page: Page) -> None:
    """Wait until the prediction metrics and the SHAP chart have rendered."""
    page.get_by_text("Model output").first.wait_for(timeout=120_000)
    page.locator("[data-testid='stImage'], [data-testid='stPyplot'] img, img").first.wait_for(timeout=120_000)
    page.wait_for_timeout(1500)


def load_preset(page: Page, label: str) -> None:
    """Pick a preset in the sidebar selectbox and apply it."""
    page.locator("[data-testid='stSidebar'] [data-testid='stSelectbox']").first.click()
    page.get_by_role("option", name=label).click()
    page.get_by_role("button", name="Load these values").click()
    page.wait_for_timeout(2500)
    wait_for_app(page)


def main() -> None:
    """Capture the prediction view (two presets) and the performance tab."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default="chrome", choices=["chrome", "msedge"])
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with streamlit_server() as url, sync_playwright() as pw:
        browser = pw.chromium.launch(channel=args.browser, headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 1500}, device_scale_factor=1.5)
        page.goto(url)
        wait_for_app(page)
        page.screenshot(path=OUT_DIR / "app_prediction_benign.png", full_page=True)
        load_preset(page, "Typical malignant (median of malignant)")
        page.screenshot(path=OUT_DIR / "app_prediction_malignant.png", full_page=True)
        page.get_by_role("tab", name="Model performance").click()
        page.wait_for_timeout(2500)
        page.screenshot(path=OUT_DIR / "app_performance.png", full_page=True)
        browser.close()
    print(f"Screenshots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
