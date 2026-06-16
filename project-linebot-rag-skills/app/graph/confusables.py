"""detect_confusables node — pg_trgm 找字形相似的韓文詞。

與主 retrieval 並行執行。結果寫入 state["confusable_words"]。
"""
from __future__ import annotations

import logging
from typing import Any

from app.graph.state import RAGState
from app.observability.tracer import traced

logger = logging.getLogger(__name__)

SIMILARITY_THRESHOLD = 0.5
MAX_RESULTS = 3


@traced("detect_confusables")
async def detect_confusables_node(state: RAGState, services: Any) -> dict[str, Any]:
    word = state.get("extracted_word")
    if not word:
        features = state.get("features")
        if features:
            word = features.primary_topic
    if not word or len(word) > 20:
        return {"confusable_words": []}

    try:
        db = services.messages_repo._client
        rows = await db.rpc(
            "find_similar_words",
            {
                "query_word": word,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "max_results": MAX_RESULTS,
            },
        )
        confusables = [
            {"title": r["title"], "content": r["content"], "sim": str(r["sim"])}
            for r in (rows or [])
        ]
    except Exception:
        logger.warning("detect_confusables failed, skipping", exc_info=True)
        confusables = []

    return {"confusable_words": confusables}
