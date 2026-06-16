import pytest
from unittest.mock import AsyncMock, MagicMock

from app.graph.confusables import detect_confusables_node
from app.graph.state import RAGState


def make_state(**kwargs) -> RAGState:
    state: RAGState = {}  # type: ignore[assignment]
    state.update(kwargs)
    return state


@pytest.mark.asyncio
async def test_returns_similar_words():
    services = MagicMock()
    services.db.rpc = AsyncMock(return_value=[
        {"title": "모레", "content": "後天", "sim": 0.75},
    ])
    state = make_state(extracted_word="모래")
    result = await detect_confusables_node(state, services)
    assert len(result["confusable_words"]) == 1
    assert result["confusable_words"][0]["title"] == "모레"


@pytest.mark.asyncio
async def test_empty_when_no_word():
    services = MagicMock()
    services.db.rpc = AsyncMock(return_value=[])
    state = make_state()
    result = await detect_confusables_node(state, services)
    assert result["confusable_words"] == []


@pytest.mark.asyncio
async def test_rpc_failure_returns_empty():
    services = MagicMock()
    services.db.rpc = AsyncMock(side_effect=Exception("connection error"))
    state = make_state(extracted_word="모래")
    result = await detect_confusables_node(state, services)
    assert result["confusable_words"] == []
