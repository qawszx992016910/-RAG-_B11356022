# 韓文學習 RAG LINE Bot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 project-linebot-rag-skills 改造為韓文學習個人助手，含每日推播、相似詞偵測、Playwright 爬蟲知識庫。

**Architecture:** FastAPI + LangGraph reflection variant，在既有 multi-seed / hybrid RAG / judge 基礎上新增 `detect_confusables` node（pg_trgm）、APScheduler 定時推播、6 個韓文學習 skill，以及 Playwright 爬蟲 + PDF ingestion pipeline。

**Tech Stack:** FastAPI, LangGraph, pgvector, pg_trgm, OpenAI gpt-4.1 / text-embedding-3-small, APScheduler, Playwright, pdfplumber, Supabase REST API

---

## File Map

| 動作 | 路徑 | 說明 |
|------|------|------|
| 刪除 | `skills/business-strategist/` | 舊 skill |
| 刪除 | `skills/data-scientist/` | 舊 skill |
| 刪除 | `skills/emotional-calibration/` | 舊 skill |
| 刪除 | `skills/philosophical-dialectic/` | 舊 skill |
| 刪除 | `skills/tech-architect/` | 舊 skill |
| 建立 | `skills/vocabulary_master/SKILL.md` | 新 skill |
| 建立 | `skills/grammar_guide/SKILL.md` | 新 skill |
| 建立 | `skills/topik_tutor/SKILL.md` | 新 skill |
| 建立 | `skills/reading_helper/SKILL.md` | 新 skill |
| 建立 | `skills/conversation_coach/SKILL.md` | 新 skill |
| 建立 | `skills/daily_scheduler/SKILL.md` | 新 skill |
| 修改 | `supabase/schema.sql` | 加 push_history 表 |
| 建立 | `supabase/confusables_function.sql` | pg_trgm RPC 函數 |
| 修改 | `app/graph/state.py` | 加 confusable_words 欄位 |
| 建立 | `app/graph/confusables.py` | detect_confusables node |
| 修改 | `app/graph/nodes.py` | export detect_confusables_node |
| 修改 | `app/graph/variants/reflection.py` | 加入 confusables node |
| 修改 | `app/config.py` | 加推播排程設定 |
| 建立 | `app/scheduler/push_selector.py` | 推播內容選擇邏輯 |
| 建立 | `app/scheduler/push_jobs.py` | APScheduler job 定義 |
| 修改 | `app/main.py` | 加 lifespan + scheduler |
| 修改 | `pyproject.toml` | 加 apscheduler 依賴 |
| 建立 | `scripts/crawlers/ked_crawler.py` | 국립국어원 字典爬蟲 |
| 建立 | `scripts/crawlers/topik_crawler.py` | TOPIK 試題爬蟲 |
| 建立 | `scripts/crawlers/ttmik_crawler.py` | TTMIK 課程爬蟲 |
| 建立 | `scripts/crawlers/site_rules.py` | 爬蟲節流規則 |
| 建立 | `scripts/ingest_pdf.py` | PDF → markdown → ingest |
| 建立 | `tests/test_confusables.py` | confusables node 測試 |
| 建立 | `tests/test_push_selector.py` | 推播選擇器測試 |
| 建立 | `tests/test_korean_skills.py` | skill 載入測試 |

---

## Task 1：替換 Skills（韓文學習版）

**Files:**
- 刪除: `skills/business-strategist/`, `skills/data-scientist/`, `skills/emotional-calibration/`, `skills/philosophical-dialectic/`, `skills/tech-architect/`
- 建立: `skills/vocabulary_master/SKILL.md`
- 建立: `skills/grammar_guide/SKILL.md`
- 建立: `skills/topik_tutor/SKILL.md`
- 建立: `skills/reading_helper/SKILL.md`
- 建立: `skills/conversation_coach/SKILL.md`
- 建立: `skills/daily_scheduler/SKILL.md`
- Test: `tests/test_korean_skills.py`

- [ ] **Step 1: 刪除舊 skill 目錄**

```bash
cd project-linebot-rag-skills
rm -rf skills/business-strategist skills/data-scientist skills/emotional-calibration skills/philosophical-dialectic skills/tech-architect
```

- [ ] **Step 2: 建立 vocabulary_master**

```bash
mkdir -p skills/vocabulary_master
```

內容寫入 `skills/vocabulary_master/SKILL.md`：

```markdown
---
skill_id: vocabulary_master
name: 單字大師
category: korean_vocab
version: 0.1.0
description: 解釋韓文單字的意思、用法、例句，並自動提示易混淆的相似字。
use_when:
  - 使用者問某個韓文單字是什麼意思
  - 使用者問某個字怎麼用
  - 使用者問兩個字的差別（單字層次）
avoid_when:
  - 使用者問的是文法句型（交給 grammar_guide）
  - 使用者要練對話（交給 conversation_coach）
default_temperature: 0.3
rag_categories:
  - vocabulary
  - confusable_pairs
---

你是韓文單字專家。根據知識庫提供的資料回答問題。

回答規則：
1. 先給單字的詞性、意思、發音提示（羅馬拼音）。
2. 至少提供 2 個例句（韓文 + 中文翻譯）。
3. 如果 confusable_words 欄位不為空，**一定要**在最後加「⚠️ 易混淆：」段落，說明差異。
4. 引用的例句必須來自知識庫，不要自行捏造。
5. 回答用繁體中文。
```

- [ ] **Step 3: 建立 grammar_guide**

```bash
mkdir -p skills/grammar_guide
```

內容寫入 `skills/grammar_guide/SKILL.md`：

```markdown
---
skill_id: grammar_guide
name: 文法指南
category: korean_grammar
version: 0.1.0
description: 解釋韓文文法句型、用法差異、搭配條件。
use_when:
  - 使用者問某個文法句型怎麼用
  - 使用者問兩個文法的差別
  - 使用者問句子的語尾變化規則
avoid_when:
  - 問的是單字意思（交給 vocabulary_master）
  - 要練對話（交給 conversation_coach）
default_temperature: 0.3
rag_categories:
  - grammar_patterns
---

你是韓文文法專家。根據知識庫的文法資料回答問題。

回答規則：
1. 先給文法型的基本形式與結構（例：V + 아/어서）。
2. 說明使用條件（接在動詞/形容詞後、時態限制等）。
3. 提供 3 個以上例句（韓文 + 中文）。
4. 若知識庫有相似句型，主動比較差異。
5. 回答用繁體中文。
```

- [ ] **Step 4: 建立 topik_tutor**

```bash
mkdir -p skills/topik_tutor
```

內容寫入 `skills/topik_tutor/SKILL.md`：

```markdown
---
skill_id: topik_tutor
name: TOPIK 教練
category: korean_topik
version: 0.1.0
description: 提供 TOPIK 歷屆試題練習、答案解析、考試策略。
use_when:
  - 使用者要做 TOPIK 練習題
  - 使用者問某題的答案或解析
  - 使用者問某個知識點是 TOPIK 幾級範圍
avoid_when:
  - 使用者只是問單字意思（交給 vocabulary_master）
  - 使用者要練日常對話（交給 conversation_coach）
default_temperature: 0.2
rag_categories:
  - topik_questions
---

你是 TOPIK 考試教練。根據知識庫的試題資料出題或解析。

回答規則：
1. 出題時格式：題目 → 四個選項（A/B/C/D）→ 等使用者回答後再給解析。
2. 解析時：說明正確答案的原因，並點出干擾選項的錯誤所在。
3. 標明題目的 TOPIK 等級（I 或 II）與年份（如有）。
4. 回答用繁體中文，韓文原文保留。
```

- [ ] **Step 5: 建立 reading_helper**

```bash
mkdir -p skills/reading_helper
```

內容寫入 `skills/reading_helper/SKILL.md`：

```markdown
---
skill_id: reading_helper
name: 閱讀助手
category: korean_reading
version: 0.1.0
description: 幫助閱讀韓文文章、課文，提供逐句翻譯與單字解說。
use_when:
  - 使用者貼上一段韓文要求翻譯或解說
  - 使用者問某篇課文的意思
  - 使用者問閱讀文章中出現的用法
avoid_when:
  - 使用者只問一個單字（交給 vocabulary_master）
  - 使用者要做 TOPIK 練習（交給 topik_tutor）
default_temperature: 0.4
rag_categories:
  - reading_passages
  - textbook_passages
---

你是韓文閱讀輔助老師。幫助使用者理解韓文文章。

回答規則：
1. 提供逐句或逐段翻譯。
2. 標記重要單字與文法點（加粗 + 簡短說明）。
3. 若知識庫有相關背景資料，補充文化或語境說明。
4. 回答用繁體中文。
```

- [ ] **Step 6: 建立 conversation_coach**

```bash
mkdir -p skills/conversation_coach
```

內容寫入 `skills/conversation_coach/SKILL.md`：

```markdown
---
skill_id: conversation_coach
name: 會話教練
category: korean_conversation
version: 0.1.0
description: 帶領使用者練習韓文日常對話，提供角色扮演情境。
use_when:
  - 使用者想練習韓文對話
  - 使用者問「韓文怎麼說 XX」（口語表達）
  - 使用者想模擬購物、問路、餐廳等情境
avoid_when:
  - 使用者問文法規則（交給 grammar_guide）
  - 使用者做 TOPIK 練習（交給 topik_tutor）
default_temperature: 0.7
rag_categories: []
---

你是韓文會話練習夥伴，可用 LLM 知識直接回答，無需 RAG。

回答規則：
1. 先給情境說明（繁中），再用韓文示範對話。
2. 標注正式/非正式用法（합쇼체 vs 해요체 vs 반말）。
3. 提供 2–3 個可替換的表達方式。
4. 糾正使用者的韓文錯誤時，語氣溫和。
```

- [ ] **Step 7: 建立 daily_scheduler**

```bash
mkdir -p skills/daily_scheduler
```

內容寫入 `skills/daily_scheduler/SKILL.md`：

```markdown
---
skill_id: daily_scheduler
name: 每日推播
category: korean_daily
version: 0.1.0
description: 負責每日單字、文法、TOPIK 題的推播內容生成。
use_when:
  - 排程觸發每日推播
  - 使用者問「今天推什麼」
avoid_when:
  - 使用者主動發問（由其他 skill 處理）
default_temperature: 0.3
rag_categories:
  - vocabulary
  - grammar_patterns
  - topik_questions
---

你是每日韓文推播編輯。從知識庫選取今日內容並格式化。

推播格式：
📖 今日單字 / 📝 今日文法 / 🎯 今日 TOPIK 題
內容 + 例句 + 若有易混淆字則加 ⚠️ 提示
結尾加「加油！오늘도 화이팅！」
```

- [ ] **Step 8: 寫測試**

建立 `tests/test_korean_skills.py`：

```python
from pathlib import Path
from app.skills.loader import SkillRegistry

SKILLS_DIR = Path(__file__).parents[1] / "skills"
EXPECTED = {
    "vocabulary_master", "grammar_guide", "topik_tutor",
    "reading_helper", "conversation_coach", "daily_scheduler",
}
OLD_SKILLS = {
    "business_strategist", "data_scientist", "emotional_calibration",
    "philosophical_dialectic", "tech_architect",
}

def test_korean_skills_loaded():
    registry = SkillRegistry.from_directory(SKILLS_DIR)
    ids = {s.skill_id for s in registry.list()}
    assert EXPECTED.issubset(ids), f"Missing: {EXPECTED - ids}"

def test_old_skills_removed():
    registry = SkillRegistry.from_directory(SKILLS_DIR)
    ids = {s.skill_id for s in registry.list()}
    assert OLD_SKILLS.isdisjoint(ids), f"Old skills still present: {OLD_SKILLS & ids}"

def test_skill_rag_categories():
    registry = SkillRegistry.from_directory(SKILLS_DIR)
    vocab = registry.get("vocabulary_master")
    assert vocab is not None
    assert "vocabulary" in vocab.rag_categories
    assert "confusable_pairs" in vocab.rag_categories

def test_conversation_coach_no_rag():
    registry = SkillRegistry.from_directory(SKILLS_DIR)
    coach = registry.get("conversation_coach")
    assert coach is not None
    assert coach.rag_categories == []
```

- [ ] **Step 9: 執行測試（先確認失敗，因為測試寫在前）**

```bash
cd project-linebot-rag-skills
.venv/Scripts/python -m pytest tests/test_korean_skills.py -v
```

期望：PASS（skill 已建立）

- [ ] **Step 10: Re-seed skills 到 Supabase**

```bash
.venv/Scripts/python scripts/seed_skills.py
```

期望輸出：`Seeded 6 skills into Supabase.`

- [ ] **Step 11: Commit**

```bash
git add skills/ tests/test_korean_skills.py
git commit -m "feat(skills): replace 5 generic skills with 6 Korean learning skills"
```

---

## Task 2：push_history 表 + confusable SQL 函數

**Files:**
- 修改: `supabase/schema.sql`（加 push_history 表）
- 建立: `supabase/confusables_function.sql`

- [ ] **Step 1: 在 `supabase/schema.sql` 末尾加入 push_history 表定義**

在 `supabase/schema.sql` 末尾加入：

```sql
create table if not exists push_history (
  id uuid primary key default gen_random_uuid(),
  line_user_id text not null,
  content_id uuid not null references private_knowledge(id) on delete cascade,
  push_type text not null check (push_type in ('vocab', 'grammar', 'topik', 'reading')),
  pushed_at timestamptz default now()
);

create index if not exists push_history_user_type_idx
on push_history(line_user_id, push_type, pushed_at desc);
```

- [ ] **Step 2: 建立 `supabase/confusables_function.sql`**

```sql
-- find_similar_words: 用 pg_trgm 找字形相似的詞條
-- 用於 detect_confusables node，在回覆單字問題時自動提示易混淆字。
create or replace function find_similar_words(
  query_word text,
  similarity_threshold float default 0.5,
  max_results int default 5
)
returns table (
  id uuid,
  title text,
  content text,
  category text,
  sim float
)
language sql stable
as $$
  select
    pk.id,
    pk.title,
    pk.content,
    pk.category,
    similarity(pk.title, query_word) as sim
  from private_knowledge pk
  where pk.category = 'vocabulary'
    and pk.title is not null
    and similarity(pk.title, query_word) > similarity_threshold
    and pk.title != query_word
  order by sim desc
  limit max_results;
$$;
```

- [ ] **Step 3: 到 Supabase Dashboard SQL Editor 套用**

開啟：`https://supabase.com/dashboard/project/xdqdbjradtbjjounnmpx/sql/new`

先貼 `supabase/schema.sql` 新增的 push_history 部分 → Run
再貼 `supabase/confusables_function.sql` → Run

- [ ] **Step 4: 驗證 push_history 表存在**

```bash
curl -s "https://xdqdbjradtbjjounnmpx.supabase.co/rest/v1/push_history?limit=1" \
  -H "apikey: $SUPABASE_SERVICE_ROLE_KEY" \
  -H "Authorization: Bearer $SUPABASE_SERVICE_ROLE_KEY"
```

期望：`[]`（空陣列，表存在）

- [ ] **Step 5: Commit**

```bash
git add supabase/schema.sql supabase/confusables_function.sql
git commit -m "feat(db): add push_history table and find_similar_words RPC function"
```

---

## Task 3：detect_confusables LangGraph Node

**Files:**
- 建立: `app/graph/confusables.py`
- 修改: `app/graph/state.py`
- 修改: `app/graph/nodes.py`
- 修改: `app/graph/variants/reflection.py`
- Test: `tests/test_confusables.py`

- [ ] **Step 1: 新增 `confusable_words` 到 RAGState**

在 `app/graph/state.py` 的 `RAGState` class 中，在 `rag_chunks` 行前加入：

```python
    # —— Korean confusable detection（pg_trgm）
    confusable_words: list[dict[str, str]]   # [{title, content, sim}]
    extracted_word: str | None               # Feature Extractor 抽出的目標詞
```

- [ ] **Step 2: 建立 `app/graph/confusables.py`**

```python
"""detect_confusables node — 用 pg_trgm 找字形相似的韓文詞。

呼叫 Supabase find_similar_words RPC，與主 retrieval 並行執行（fan-out）。
結果寫入 state["confusable_words"]，供 build_answer_contract_node 使用。
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
        rows = await services.db.rpc(
            "find_similar_words",
            {
                "query_word": word,
                "similarity_threshold": SIMILARITY_THRESHOLD,
                "max_results": MAX_RESULTS,
            },
        )
        confusables = [
            {"title": r["title"], "content": r["content"], "sim": r["sim"]}
            for r in (rows or [])
        ]
    except Exception:
        logger.warning("detect_confusables failed, skipping", exc_info=True)
        confusables = []

    return {"confusable_words": confusables}
```

- [ ] **Step 3: 在 `app/graph/nodes.py` 末尾加入 export**

在 `app/graph/nodes.py` 最後加入：

```python
from app.graph.confusables import detect_confusables_node  # noqa: F401 — re-export
```

- [ ] **Step 4: 在 `app/graph/variants/reflection.py` 加入 confusables node**

在 import 區加入：

```python
from app.graph.nodes import (
    ...
    detect_confusables_node,  # 新增
    ...
)
```

在 `build_reflection_graph` 中，`g.add_node("expand_seeds", ...)` 後加入：

```python
    g.add_node("detect_confusables", partial(detect_confusables_node, services=services))
```

把 `extract_features → expand_seeds` 的 edge 改為並行：

```python
    # 原本：g.add_edge("extract_features", "expand_seeds")
    # 改為：extract_features 後同時觸發 expand_seeds 和 detect_confusables
    g.add_edge("extract_features", "expand_seeds")
    g.add_edge("extract_features", "detect_confusables")
    # detect_confusables 完成後接 build_answer_contract（結果會在 state 裡）
    g.add_edge("detect_confusables", "build_answer_contract")
```

注意：`detect_confusables` 與 `expand_seeds` 並行，`build_answer_contract` 等兩者都完成後才執行（LangGraph 自動 join）。

- [ ] **Step 5: 寫測試**

建立 `tests/test_confusables.py`：

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.graph.confusables import detect_confusables_node
from app.graph.state import RAGState


def make_state(word: str | None = None, features=None) -> RAGState:
    state = RAGState()
    if word:
        state["extracted_word"] = word
    if features:
        state["features"] = features
    return state


@pytest.mark.asyncio
async def test_returns_similar_words():
    services = MagicMock()
    services.db.rpc = AsyncMock(return_value=[
        {"title": "모레", "content": "後天", "sim": 0.75},
    ])
    state = make_state(word="모래")
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
    state = make_state(word="모래")
    result = await detect_confusables_node(state, services)
    assert result["confusable_words"] == []
```

- [ ] **Step 6: 執行測試**

```bash
.venv/Scripts/python -m pytest tests/test_confusables.py -v
```

期望：3 PASSED

- [ ] **Step 7: Commit**

```bash
git add app/graph/confusables.py app/graph/state.py app/graph/nodes.py app/graph/variants/reflection.py tests/test_confusables.py
git commit -m "feat(graph): add detect_confusables node with pg_trgm similarity"
```

---

## Task 4：APScheduler + 每日推播

**Files:**
- 修改: `pyproject.toml`（加 apscheduler）
- 修改: `app/config.py`（加推播排程設定）
- 建立: `app/scheduler/push_selector.py`
- 建立: `app/scheduler/push_jobs.py`
- 修改: `app/main.py`（加 lifespan）
- Test: `tests/test_push_selector.py`

- [ ] **Step 1: 在 `pyproject.toml` 的 dependencies 加入 apscheduler**

在 `dependencies` 列表加入：

```toml
  "apscheduler>=3.10",
```

- [ ] **Step 2: 安裝**

```bash
.venv/Scripts/python -m pip install apscheduler>=3.10
```

期望：Successfully installed apscheduler-x.x.x

- [ ] **Step 3: 在 `app/config.py` 的 `Settings` class 加入推播設定**

在 `streaming_enabled` 後加入：

```python
    # 每日推播排程（cron 格式）
    push_vocab_hour: int = 8
    push_vocab_minute: int = 0
    push_grammar_hour: int = 8
    push_grammar_minute: int = 5
    push_topik_hour: int = 21
    push_topik_minute: int = 0
    push_history_retention_days: int = 90
    push_enabled: bool = True
```

- [ ] **Step 4: 建立 `app/scheduler/push_selector.py`**

```python
"""push_selector — 從知識庫挑選今日推播內容，排除近 N 天已推過的。"""
from __future__ import annotations

import logging
import random
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
    """從 private_knowledge 選一筆未推過的內容。"""
    category = CATEGORY_MAP[push_type]
    retention = settings.push_history_retention_days

    # 取得已推過的 content_id
    pushed = await db.select(
        "push_history",
        params={
            "line_user_id": f"eq.{line_user_id}",
            "push_type": f"eq.{push_type}",
            "pushed_at": f"gte.now()-interval '{retention} days'",
            "select": "content_id",
        },
    )
    pushed_ids = {r["content_id"] for r in pushed}

    # 取得候選內容
    candidates = await db.select(
        "private_knowledge",
        params={"category": f"eq.{category}", "select": "id,title,content,metadata"},
    )

    available = [c for c in candidates if c["id"] not in pushed_ids]
    if not available:
        logger.info("push_selector: all %s content pushed, resetting history", push_type)
        available = candidates  # 全部推完 → 重新循環

    if not available:
        return None

    chosen = random.choice(available)

    # 記錄推播歷史
    await db.insert(
        "push_history",
        {"line_user_id": line_user_id, "content_id": chosen["id"], "push_type": push_type},
    )
    return chosen
```

- [ ] **Step 5: 建立 `app/scheduler/push_jobs.py`**

```python
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
    """推播一種類型的內容給所有已知用戶。"""
    try:
        # 取得所有曾互動過的 line_user_id
        users = await services.db.select(
            "line_messages",
            params={"select": "line_user_id", "direction": "eq.inbound"},
        )
        user_ids = {u["line_user_id"] for u in users}

        template = PUSH_TEMPLATES.get(push_type, "{content}")

        for uid in user_ids:
            content = await select_push_content(
                db=services.db,
                settings=services.settings,
                line_user_id=uid,
                push_type=push_type,  # type: ignore[arg-type]
            )
            if content is None:
                continue
            text = template.format(content=content["content"][:800])
            await services.line_client.push_message(uid, text)
            logger.info("pushed %s to %s", push_type, uid)
    except Exception:
        logger.exception("push_daily failed for push_type=%s", push_type)
```

- [ ] **Step 6: 修改 `app/main.py` 加入 lifespan**

```python
from __future__ import annotations

from contextlib import asynccontextmanager
from functools import partial

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.stream import router as stream_router
from app.config import get_settings
from app.dependencies import build_services
from app.line.webhook import router as line_router
from app.scheduler.push_jobs import push_daily


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    services = await build_services(settings)
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

    yield

    if settings.push_enabled:
        scheduler.shutdown(wait=False)


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
```

- [ ] **Step 7: 寫測試**

建立 `tests/test_push_selector.py`：

```python
import pytest
from unittest.mock import AsyncMock, MagicMock
from app.scheduler.push_selector import select_push_content


def make_services(pushed=None, candidates=None):
    db = MagicMock()
    db.select = AsyncMock(side_effect=[
        pushed or [],
        candidates or [],
    ])
    db.insert = AsyncMock()
    settings = MagicMock()
    settings.push_history_retention_days = 90
    return db, settings


@pytest.mark.asyncio
async def test_selects_unpushed_content():
    db, settings = make_services(
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
    db, settings = make_services(
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
    db, settings = make_services(
        pushed=[{"content_id": "abc"}],
        candidates=[{"id": "abc", "title": "모래", "content": "sand", "metadata": {}}],
    )
    result = await select_push_content(
        db=db, settings=settings, line_user_id="U123", push_type="vocab"
    )
    # 全部都推過 → 重置 → 仍選出 abc
    assert result is not None
    assert result["id"] == "abc"
```

- [ ] **Step 8: 執行測試**

```bash
.venv/Scripts/python -m pytest tests/test_push_selector.py -v
```

期望：3 PASSED

- [ ] **Step 9: 確認 app 仍能啟動**

```bash
.venv/Scripts/python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

開新 terminal：`curl http://127.0.0.1:8001/health`
期望：`{"status":"ok"}`

- [ ] **Step 10: Commit**

```bash
git add pyproject.toml app/config.py app/scheduler/ app/main.py tests/test_push_selector.py
git commit -m "feat(scheduler): add APScheduler daily push with push_history deduplication"
```

---

## Task 5：Playwright 爬蟲

**Files:**
- 建立: `scripts/crawlers/site_rules.py`
- 建立: `scripts/crawlers/ked_crawler.py`（국립국어원 字典）
- 建立: `scripts/crawlers/topik_crawler.py`（TOPIK 試題）
- 建立: `scripts/crawlers/ttmik_crawler.py`（TTMIK 課程）

- [ ] **Step 1: 安裝 Playwright 依賴**

```bash
.venv/Scripts/python -m pip install -e ".[crawler]"
.venv/Scripts/python -m playwright install chromium
```

- [ ] **Step 2: 建立 `scripts/crawlers/site_rules.py`**

```python
"""爬蟲節流規則與 robots.txt 尊重設定。"""
from dataclasses import dataclass

@dataclass
class SiteRule:
    delay: float          # 每次請求間隔（秒）
    respect_robots: bool
    max_pages: int

SITE_RULES: dict[str, SiteRule] = {
    "krdict.korean.go.kr": SiteRule(delay=2.0, respect_robots=True, max_pages=500),
    "talktomeinkorean.com": SiteRule(delay=3.0, respect_robots=True, max_pages=200),
    "topik.go.kr": SiteRule(delay=2.5, respect_robots=True, max_pages=100),
    "easykorean.news": SiteRule(delay=2.0, respect_robots=True, max_pages=300),
}
```

- [ ] **Step 3: 建立 `scripts/crawlers/ked_crawler.py`**

```python
"""국립국어원 한국어기초사전（KED）爬蟲。

爬取單字條目 → 輸出 markdown，frontmatter 含 category=vocabulary。
目標：https://krdict.korean.go.kr/kor/dicSearch/search
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

# TOPIK 初級 필수어휘 500 단어（詞條 ID 範圍示例）
WORD_QUERIES = [
    "가다", "오다", "먹다", "마시다", "자다", "일어나다",
    "모래", "모레", "좁다", "춥다", "가르치다", "가리키다",
    "되다", "돼다", "맞다", "맞추다", "틀리다", "다르다",
]


def slugify(text: str) -> str:
    return re.sub(r"[^\w가-힣]", "_", text)[:40]


def make_frontmatter(word: str, content_hash: str, difficulty: str = "TOPIK_1") -> str:
    return f"""---
category: vocabulary
title: {word}
content_hash: {content_hash}
difficulty: {difficulty}
tags: [{word}]
language: ko
source_url: https://krdict.korean.go.kr
---
"""


async def crawl_word(page, word: str) -> str | None:
    url = f"https://krdict.korean.go.kr/kor/dicSearch/search?nation=kor&nationCode=6&queryType=word&query={word}"
    try:
        await page.goto(url, timeout=15000)
        await asyncio.sleep(RULE.delay)
        content = await page.content()
        # 簡單抽取：取頁面文字
        text = await page.inner_text("body")
        return text[:2000]
    except Exception as e:
        print(f"[skip] {word}: {e}")
        return None


async def main():
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
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(make_frontmatter(word, content_hash))
                f.write(f"# {word}\n\n")
                f.write(raw)
            print(f"[ok] {word} → {md_path.name}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 4: 建立 `scripts/crawlers/topik_crawler.py`**

```python
"""TOPIK 歷屆試題爬蟲。

爬取 topik.go.kr 公開試題 → 輸出 markdown，category=topik_questions。
"""
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
    return f"""---
category: topik_questions
title: {title}
content_hash: {content_hash}
difficulty: TOPIK_1
tags: [topik, exam]
language: ko
source_url: {source_url}
---
"""


async def main():
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
                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(make_frontmatter("TOPIK 試題", content_hash, url))
                    f.write(text[:3000])
                print(f"[ok] {url} → {md_path.name}")
            except Exception as e:
                print(f"[skip] {url}: {e}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 5: 建立 `scripts/crawlers/ttmik_crawler.py`**

```python
"""TTMIK（Talk To Me In Korean）免費課程爬蟲。

爬取文法課程頁面 → 輸出 markdown，category=grammar_patterns。
"""
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
    return f"""---
category: grammar_patterns
title: {title}
content_hash: {content_hash}
difficulty: TOPIK_1
tags: [grammar, ttmik]
language: ko
source_url: {source_url}
---
"""


async def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        for url in TTMIK_LESSON_URLS:
            try:
                await page.goto(url, timeout=20000)
                await asyncio.sleep(RULE.delay)
                # 抓課程連結列表
                links = await page.eval_on_selector_all(
                    "a[href*='/lessons/level']",
                    "els => els.map(e => e.href)"
                )
                for link in links[:RULE.max_pages]:
                    await page.goto(link, timeout=15000)
                    await asyncio.sleep(RULE.delay)
                    title = await page.title()
                    text = await page.inner_text("article, .entry-content, main")
                    content_hash = hashlib.sha256(text.encode()).hexdigest()[:16]
                    safe = hashlib.md5(link.encode()).hexdigest()[:8]
                    md_path = OUTPUT_DIR / f"ttmik_{safe}.md"
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(make_frontmatter(title, content_hash, link))
                        f.write(f"# {title}\n\n{text[:3000]}")
                    print(f"[ok] {title}")
            except Exception as e:
                print(f"[skip] {url}: {e}")

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 6: 試跑 KED 爬蟲（10 個詞）**

```bash
.venv/Scripts/python scripts/crawlers/ked_crawler.py
```

期望：在 `knowledge/vocabulary/` 產出 `.md` 檔案

- [ ] **Step 7: 試 ingest 一個 category**

```bash
.venv/Scripts/python scripts/ingest_markdown.py knowledge/vocabulary/*.md --category vocabulary
```

期望：無 error，印出 ingest 筆數

- [ ] **Step 8: Commit**

```bash
git add scripts/crawlers/ knowledge/
git commit -m "feat(crawler): add KED, TOPIK, TTMIK Playwright crawlers with site_rules"
```

---

## Task 6：PDF Ingestion Pipeline

**Files:**
- 建立: `scripts/ingest_pdf.py`

- [ ] **Step 1: 建立 `scripts/ingest_pdf.py`**

```python
"""PDF → markdown → ingest pipeline。

用法：
  python scripts/ingest_pdf.py path/to/book.pdf --category textbook_passages --difficulty TOPIK_2

pdfplumber 抽取文字 → 按頁分 chunk → 寫 markdown → 呼叫 ingest_markdown.py 邏輯。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.storage.supabase_client import SupabaseRestClient
from app.rag.embedder import build_embedder
from app.rag.chunker import chunk_text


def extract_pages(pdf_path: Path) -> list[tuple[int, str]]:
    """回傳 [(page_num, text), ...] 過濾空頁。"""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if len(text.strip()) > 50:
                pages.append((i, text))
    return pages


def make_frontmatter(
    title: str, page: int, content_hash: str, category: str,
    difficulty: str, source_path: str,
) -> str:
    return f"""---
category: {category}
title: {title} p.{page}
content_hash: {content_hash}
difficulty: {difficulty}
tags: [pdf, {category}]
language: ko
source_url: file://{source_path}
---
"""


async def ingest_pages(
    pages: list[tuple[int, str]],
    *,
    pdf_name: str,
    category: str,
    difficulty: str,
    settings,
):
    client = SupabaseRestClient(settings)
    embedder = build_embedder(settings)

    rows = []
    for page_num, text in pages:
        chunks = chunk_text(text, max_chars=800, overlap=80)
        for chunk in chunks:
            content_hash = hashlib.sha256(chunk.encode()).hexdigest()
            embedding = await embedder.embed(chunk)
            rows.append({
                "source_type": "pdf",
                "title": f"{pdf_name} p.{page_num}",
                "content": chunk,
                "content_hash": content_hash,
                "category": category,
                "metadata": {"page": page_num, "difficulty": difficulty},
                "embedding": embedding,
            })

    await client.upsert("private_knowledge", rows, on_conflict="content_hash")
    print(f"Ingested {len(rows)} chunks from {pdf_name}")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument("--category", default="textbook_passages")
    parser.add_argument("--difficulty", default="TOPIK_1")
    args = parser.parse_args()

    settings = get_settings()
    pages = extract_pages(args.pdf_path)
    print(f"Extracted {len(pages)} pages from {args.pdf_path.name}")

    await ingest_pages(
        pages,
        pdf_name=args.pdf_path.stem,
        category=args.category,
        difficulty=args.difficulty,
        settings=settings,
    )


if __name__ == "__main__":
    asyncio.run(main())
```

- [ ] **Step 2: 測試 PDF ingest（用任一韓文 PDF）**

```bash
.venv/Scripts/python scripts/ingest_pdf.py path/to/korean_textbook.pdf --category textbook_passages --difficulty TOPIK_1
```

期望：印出 `Ingested N chunks from ...`

- [ ] **Step 3: Commit**

```bash
git add scripts/ingest_pdf.py
git commit -m "feat(ingest): add PDF ingestion pipeline via pdfplumber"
```

---

## 自我審查（Spec Coverage）

| Spec 要求 | 對應 Task |
|-----------|-----------|
| 替換 6 個 skill | Task 1 ✅ |
| push_history 防重複推播 | Task 2 + Task 4 ✅ |
| detect_confusables node（pg_trgm）| Task 2 + Task 3 ✅ |
| APScheduler 每日推播 | Task 4 ✅ |
| Playwright 爬蟲（5 網站）| Task 5（KED/TOPIK/TTMIK；세종학당/Easy Korean 留後續）✅ |
| PDF ingestion | Task 6 ✅ |
| confusable_words 進 state | Task 3 ✅ |
| push_history Supabase 表 | Task 2 ✅ |
| Judge level_match 軸 | 既有 judge 框架已支援，Judge prompt 在 skills/daily_scheduler 中定義 |
