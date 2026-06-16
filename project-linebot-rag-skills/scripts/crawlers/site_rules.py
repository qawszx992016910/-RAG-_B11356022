"""爬蟲節流規則與 robots.txt 尊重設定。"""
from dataclasses import dataclass


@dataclass
class SiteRule:
    delay: float
    respect_robots: bool
    max_pages: int


SITE_RULES: dict[str, SiteRule] = {
    "krdict.korean.go.kr": SiteRule(delay=2.0, respect_robots=True, max_pages=500),
    "talktomeinkorean.com": SiteRule(delay=3.0, respect_robots=True, max_pages=200),
    "topik.go.kr": SiteRule(delay=2.5, respect_robots=True, max_pages=100),
    "easykorean.news": SiteRule(delay=2.0, respect_robots=True, max_pages=300),
}
