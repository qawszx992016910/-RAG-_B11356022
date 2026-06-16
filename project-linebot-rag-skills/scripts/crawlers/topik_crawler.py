"""TOPIK 歷屆試題爬蟲。"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.async_api import async_playwright

from scripts.crawlers.site_rules import SITE_RULES

OUTPUT_DIR = PROJECT_ROOT / "knowledge" / "topik_questions"
RULE = SITE_RULES["topik.go.kr"]

TOPIK_EXAM_URLS = [
    "https://www.topik.go.kr/usr/cmm/subLocation.do?menuSeq=2110503",
]


def make_frontmatter(title: str, content_hash: str, source_url: str) -> str:
    return (
        "---\n"
        f"category: topik_questions\n"
        f"title: {title}\n"
        f"content_hash: {content_hash}\n"
        f"difficulty: TOPIK_1\n"
        f"tags: [topik, exam]\n"
        f"language: ko\n"
        f"source_url: {source_url}\n"
        "---\n"
    )


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        for url in TOPIK_EXAM_URLS:
            try:
                await page.goto(url, timeout=20000)
                await asyncio.sleep(RULE.delay)
                text = await page.inner_text("body")
                content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                md_path = OUTPUT_DIR / f"topik_{content_hash}.md"
                md_path.write_text(
                    make_frontmatter("TOPIK 試題", content_hash, url) + text[:3000],
                    encoding="utf-8",
                )
                print(f"[ok] {url} → {md_path.name}")
            except Exception as e:
                print(f"[skip] {url}: {e}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
