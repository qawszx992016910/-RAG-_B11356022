import pytest
from unittest.mock import AsyncMock, MagicMock

from app.scheduler.push_selector import select_push_content


def make_db_and_settings(pushed=None, candidates=None):
    db = MagicMock()
    db.select = AsyncMock(side_effect=[
        pushed if pushed is not None else [],
        candidates if candidates is not None else [],
    ])
    db.insert = AsyncMock()
    settings = MagicMock()
    settings.push_history_retention_days = 90
    return db, settings


@pytest.mark.asyncio
async def test_selects_unpushed_content():
    db, settings = make_db_and_settings(
        pushed=[],
        candidates=[{"id": "abc", "title": "모래", "content": "sand", "metadata": {}}],
    )
    result = await select_push_content(
        db=db, settings=settings, line_user_id="U123", push_type="vocab"
    )
    assert result is not None
    assert result["id"] == "abc"
    db.insert.assert_called_once()


@pytest.mark.asyncio
async def test_skips_already_pushed():
    db, settings = make_db_and_settings(
        pushed=[{"content_id": "abc"}],
        candidates=[
            {"id": "abc", "title": "모래", "content": "sand", "metadata": {}},
            {"id": "def", "title": "모레", "content": "day after tomorrow", "metadata": {}},
        ],
    )
    result = await select_push_content(
        db=db, settings=settings, line_user_id="U123", push_type="vocab"
    )
    assert result is not None
    assert result["id"] == "def"


@pytest.mark.asyncio
async def test_resets_when_all_pushed():
    db, settings = make_db_and_settings(
        pushed=[{"content_id": "abc"}],
        candidates=[{"id": "abc", "title": "모래", "content": "sand", "metadata": {}}],
    )
    result = await select_push_content(
        db=db, settings=settings, line_user_id="U123", push_type="vocab"
    )
    assert result is not None
    assert result["id"] == "abc"
