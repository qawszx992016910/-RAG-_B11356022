"""국립국어원 한국어기초사전（KED）爬蟲。

爬取單字條目 → 輸出 markdown，frontmatter 含 category=vocabulary。
"""
from __future__ import annotations

import asyncio
import hashlib
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from playwright.async_api import async_playwright

from scripts.crawlers.site_rules import SITE_RULES

OUTPUT_DIR = PROJECT_ROOT / "knowledge" / "vocabulary"
RULE = SITE_RULES["krdict.korean.go.kr"]

WORD_QUERIES = [
    "가다", "오다", "먹다", "마시다", "자다", "일어나다",
    "모래", "모레", "좁다", "춥다", "가르치다", "가리키다",
    "되다", "맞다", "맞추다", "틀리다", "다르다",
]


def slugify(text: str) -> str:
    return re.sub(r"[^\w가-힣]", "_", text)[:40]


def make_frontmatter(word: str, content_hash: str, difficulty: str = "TOPIK_1") -> str:
    return (
        "---\n"
        f"category: vocabulary\n"
        f"title: {word}\n"
        f"content_hash: {content_hash}\n"
        f"difficulty: {difficulty}\n"
        f"tags: [{word}]\n"
        f"language: ko\n"
        f"source_url: https://krdict.korean.go.kr\n"
        "---\n"
    )


async def crawl_word(page, word: str) -> str | None:
    url = (
        "https://krdict.korean.go.kr/kor/dicSearch/search"
        f"?nation=kor&nationCode=6&queryType=word&query={word}"
    )
    try:
        await page.goto(url, timeout=15000)
        await asyncio.sleep(RULE.delay)
        text = await page.inner_text("body")
        return text[:2000]
    except Exception as e:
        print(f"[skip] {word}: {e}")
        return None


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        for word in WORD_QUERIES[:RULE.max_pages]:
            raw = await crawl_word(page, word)
            if not raw:
                continue
            content_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
            md_path = OUTPUT_DIR / f"{slugify(word)}.md"
            md_path.write_text(
                make_frontmatter(word, content_hash) + f"# {word}\n\n{raw}",
                encoding="utf-8",
            )
            print(f"[ok] {word} → {md_path.name}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
