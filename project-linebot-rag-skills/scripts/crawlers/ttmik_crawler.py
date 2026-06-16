"""TTMIK（Talk To Me In Korean）免費課程爬蟲。"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.async_api import async_playwright

from scripts.crawlers.site_rules import SITE_RULES

OUTPUT_DIR = PROJECT_ROOT / "knowledge" / "grammar_patterns"
RULE = SITE_RULES["talktomeinkorean.com"]

TTMIK_LESSON_URLS = [
    "https://talktomeinkorean.com/lessons/",
]


def make_frontmatter(title: str, content_hash: str, source_url: str) -> str:
    return (
        "---\n"
        f"category: grammar_patterns\n"
        f"title: {title}\n"
        f"content_hash: {content_hash}\n"
        f"difficulty: TOPIK_1\n"
        f"tags: [grammar, ttmik]\n"
        f"language: ko\n"
        f"source_url: {source_url}\n"
        "---\n"
    )


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        for url in TTMIK_LESSON_URLS:
            try:
                await page.goto(url, timeout=20000)
                await asyncio.sleep(RULE.delay)
                links = await page.eval_on_selector_all(
                    "a[href*='/lessons/level']",
                    "els => els.map(e => e.href)",
                )
                for link in links[:RULE.max_pages]:
                    try:
                        await page.goto(link, timeout=15000)
                        await asyncio.sleep(RULE.delay)
                        title = await page.title()
                        text = await page.inner_text("article, .entry-content, main")
                        content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                        safe = hashlib.md5(link.encode()).hexdigest()[:8]
                        md_path = OUTPUT_DIR / f"ttmik_{safe}.md"
                        md_path.write_text(
                            make_frontmatter(title, content_hash, link) + f"# {title}\n\n{text[:3000]}",
                            encoding="utf-8",
                        )
                        print(f"[ok] {title}")
                    except Exception as e:
                        print(f"[skip] {link}: {e}")
            except Exception as e:
                print(f"[skip] {url}: {e}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
