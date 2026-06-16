from __future__ import annotations

from contextlib import asynccontextmanager
from functools import partial

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.stream import router as stream_router
from app.config import get_settings
from app.dependencies import get_runtime_services
from app.line.webhook import router as line_router
from app.scheduler.push_jobs import push_daily


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    services = get_runtime_services()
    app.state.services = services

    scheduler = AsyncIOScheduler()
    if settings.push_enabled:
        scheduler.add_job(
            partial(push_daily, services=services, push_type="vocab"),
            "cron", hour=settings.push_vocab_hour, minute=settings.push_vocab_minute,
        )
        scheduler.add_job(
            partial(push_daily, services=services, push_type="grammar"),
            "cron", hour=settings.push_grammar_hour, minute=settings.push_grammar_minute,
        )
        scheduler.add_job(
            partial(push_daily, services=services, push_type="topik"),
            "cron", hour=settings.push_topik_hour, minute=settings.push_topik_minute,
        )
        scheduler.start()
        app.state.scheduler = scheduler

    yield

    if settings.push_enabled and hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown(wait=False)


def create_app() -> FastAPI:
    app = FastAPI(title="project-linebot-rag-skills", lifespan=lifespan)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(line_router)
    app.include_router(chat_router)
    app.include_router(stream_router)
    return app


app = create_app()
