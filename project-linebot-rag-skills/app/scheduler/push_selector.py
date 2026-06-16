"""push_selector — 從知識庫挑選今日推播內容，排除近 N 天已推過的。"""
from __future__ import annotations

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Literal

logger = logging.getLogger(__name__)

PushType = Literal["vocab", "grammar", "topik", "reading"]

CATEGORY_MAP: dict[PushType, str] = {
    "vocab": "vocabulary",
    "grammar": "grammar_patterns",
    "topik": "topik_questions",
    "reading": "reading_passages",
}


async def select_push_content(
    *,
    db,
    settings,
    line_user_id: str,
    push_type: PushType,
) -> dict | None:
    """從 private_knowledge 選一筆未推過的內容。全推完則重新循環。"""
    category = CATEGORY_MAP[push_type]
    retention = settings.push_history_retention_days

    cutoff = (datetime.now(timezone.utc) - timedelta(days=retention)).isoformat()
    pushed = await db.select(
        "push_history",
        params={
            "line_user_id": f"eq.{line_user_id}",
            "push_type": f"eq.{push_type}",
            "pushed_at": f"gte.{cutoff}",
            "select": "content_id",
        },
    )
    pushed_ids = {r["content_id"] for r in pushed}

    candidates = await db.select(
        "private_knowledge",
        params={"category": f"eq.{category}", "select": "id,title,content,metadata"},
    )

    available = [c for c in candidates if c["id"] not in pushed_ids]
    if not available:
        logger.info("push_selector: all %s content pushed, resetting", push_type)
        available = candidates

    if not available:
        return None

    chosen = random.choice(available)
    await db.insert(
        "push_history",
        {"line_user_id": line_user_id, "content_id": chosen["id"], "push_type": push_type},
    )
    return chosen
