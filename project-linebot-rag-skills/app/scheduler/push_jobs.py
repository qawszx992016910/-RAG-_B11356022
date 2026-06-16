"""push_jobs — APScheduler job 函數，每日定時推播給所有訂閱用戶。"""
from __future__ import annotations

import logging

from app.scheduler.push_selector import select_push_content

logger = logging.getLogger(__name__)

PUSH_TEMPLATES: dict[str, str] = {
    "vocab": "📖 *今日單字*\n\n{content}\n\n오늘도 화이팅！",
    "grammar": "📝 *今日文法*\n\n{content}\n\n오늘도 화이팅！",
    "topik": "🎯 *今日 TOPIK 題*\n\n{content}\n\n오늘도 화이팅！",
}


async def push_daily(*, services, push_type: str) -> None:
    """推播一種類型的內容給所有曾互動過的 LINE 用戶。

    services is RuntimeServices (app.dependencies).
    - services.messages_repo._client  → SupabaseRestClient (has .select / .insert)
    - services.line_client.push_text  → send LINE push message
    - services.settings               → Settings
    """
    try:
        db = services.messages_repo._client

        users = await db.select(
            "line_messages",
            params={"select": "line_user_id", "direction": "eq.inbound"},
        )
        user_ids = {u["line_user_id"] for u in users}

        template = PUSH_TEMPLATES.get(push_type, "{content}")

        for uid in user_ids:
            content = await select_push_content(
                db=db,
                settings=services.settings,
                line_user_id=uid,
                push_type=push_type,  # type: ignore[arg-type]
            )
            if content is None:
                continue
            text = template.format(content=content["content"][:800])
            await services.line_client.push_text(uid, text)
            logger.info("pushed %s to %s", push_type, uid)
    except Exception:
        logger.exception("push_daily failed for push_type=%s", push_type)
