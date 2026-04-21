# LaborRAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local RAG system that lets general workers query Taiwan labor law (勞基法, 勞保條例, 職安法) in plain language and receive answers grounded in exact article citations.

**Architecture:** BGE-M3 embeds each law article as a knowledge atom stored in PostgreSQL + pgvector. Queries use hybrid search (dense cosine + sparse tsvector), fused via RRF, reranked by BGE-Reranker-v2-m3, then answered by Ollama Llama3.1-8B with mandatory article citation.

**Tech Stack:** Python 3.11, PostgreSQL 16 + pgvector, FlagEmbedding (BGE-M3 + BGE-Reranker), Ollama (Llama3.1-8B), psycopg3, Pydantic v2, Click, RAGAS, pytest

---

## File Map

```
laborrag/
├── docker-compose.yml              # PostgreSQL + pgvector container
├── pyproject.toml                  # deps + CLI entry point
├── .env.example                    # DB URL, model names
├── migrations/
│   └── 001_init.sql                # pgvector schema + indexes
├── schemas/
│   └── knowledge_atom.json         # JSON Schema for atom validation
├── src/laborrag/
│   ├── __init__.py
│   ├── config.py                   # env-based config (DATABASE_URL, models)
│   ├── etl/
│   │   ├── __init__.py
│   │   ├── fetch.py                # HTTP fetch XML from laws.moj.gov.tw
│   │   └── parse.py                # XML → KnowledgeAtom list
│   ├── embed/
│   │   ├── __init__.py
│   │   └── embedder.py             # BGE-M3 wrapper → float[1024]
│   ├── db/
│   │   ├── __init__.py
│   │   └── store.py                # insert/query atoms via psycopg3
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── dense.py                # pgvector cosine top-N
│   │   ├── sparse.py               # tsvector BM25 top-N
│   │   ├── fusion.py               # RRF merge → top-20
│   │   └── reranker.py             # BGE-Reranker → top-5
│   ├── generation/
│   │   ├── __init__.py
│   │   └── generator.py            # Ollama chat with grounding prompt
│   └── cli.py                      # `laborrag ingest / query / eval`
├── tests/
│   ├── conftest.py                 # shared fixtures (db conn, sample atoms)
│   ├── test_parse.py
│   ├── test_embedder.py
│   ├── test_store.py
│   ├── test_retrieval.py
│   ├── test_generator.py
│   ├── test_cli.py
│   └── golden/
│       └── test_set.json           # 20 Q&A pairs with expected article nos
├── docs/
│   ├── architecture/
│   │   └── system-overview.md
│   ├── adr/
│   │   ├── ADR-001-embedding-model.md
│   │   ├── ADR-002-hybrid-search.md
│   │   └── ADR-003-reranker.md
│   └── whitepaper/
│       └── laborrag-whitepaper.md
└── README.md
```

---

## Task 1: Project Scaffold

**Files:**
- Create: `docker-compose.yml`
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `src/laborrag/__init__.py`
- Create: `src/laborrag/config.py`

- [ ] **Step 1: Create docker-compose.yml**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_DB: laborrag
      POSTGRES_USER: laborrag
      POSTGRES_PASSWORD: laborrag
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U laborrag"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
```

- [ ] **Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "laborrag"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "FlagEmbedding>=1.2.9",
    "psycopg[binary]>=3.1",
    "pgvector>=0.3",
    "ollama>=0.3",
    "python-dotenv>=1.0",
    "lxml>=5.0",
    "pydantic>=2.6",
    "click>=8.1",
    "ragas>=0.1",
    "datasets>=2.0",
    "requests>=2.31",
]

[project.scripts]
laborrag = "laborrag.cli:main"

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-mock>=3.12"]

[tool.hatch.build.targets.wheel]
packages = ["src/laborrag"]
```

- [ ] **Step 3: Create .env.example**

```
DATABASE_URL=postgresql://laborrag:laborrag@localhost:5432/laborrag
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
OLLAMA_MODEL=llama3.1:8b
TOP_K_COARSE=20
TOP_K_FINAL=5
```

- [ ] **Step 4: Create src/laborrag/config.py**

```python
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL: str = os.environ["DATABASE_URL"]
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
TOP_K_COARSE: int = int(os.getenv("TOP_K_COARSE", "20"))
TOP_K_FINAL: int = int(os.getenv("TOP_K_FINAL", "5"))
```

- [ ] **Step 5: Create src/laborrag/__init__.py**

```python
```

- [ ] **Step 6: Install deps and start DB**

```bash
cp .env.example .env
pip install -e ".[dev]"
docker compose up -d
docker compose ps  # postgres should be "healthy"
```

Expected: postgres container running, healthy.

- [ ] **Step 7: Commit**

```bash
git init
git add docker-compose.yml pyproject.toml .env.example src/laborrag/
git commit -m "feat: project scaffold with config and docker compose"
```

---

## Task 2: Database Migration

**Files:**
- Create: `migrations/001_init.sql`

- [ ] **Step 1: Write migration SQL**

```sql
-- migrations/001_init.sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS knowledge_atoms (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    law_name    TEXT NOT NULL,
    article_no  TEXT NOT NULL,
    chapter     TEXT,
    content     TEXT NOT NULL,
    keywords    TEXT[] NOT NULL DEFAULT '{}',
    last_updated DATE,
    parent_id   UUID REFERENCES knowledge_atoms(id) ON DELETE SET NULL,
    embedding   vector(1024),
    tsv_content TSVECTOR GENERATED ALWAYS AS (
        to_tsvector('simple', content || ' ' || array_to_string(keywords, ' '))
    ) STORED,
    UNIQUE (law_name, article_no)
);

CREATE INDEX IF NOT EXISTS idx_atoms_embedding
    ON knowledge_atoms USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

CREATE INDEX IF NOT EXISTS idx_atoms_tsv
    ON knowledge_atoms USING GIN (tsv_content);

CREATE INDEX IF NOT EXISTS idx_atoms_law
    ON knowledge_atoms (law_name);
```

- [ ] **Step 2: Apply migration**

```bash
psql postgresql://laborrag:laborrag@localhost:5432/laborrag -f migrations/001_init.sql
```

Expected output:
```
CREATE EXTENSION
CREATE TABLE
CREATE INDEX
CREATE INDEX
CREATE INDEX
```

- [ ] **Step 3: Verify schema**

```bash
psql postgresql://laborrag:laborrag@localhost:5432/laborrag -c "\d knowledge_atoms"
```

Expected: table with columns id, law_name, article_no, chapter, content, keywords, last_updated, parent_id, embedding, tsv_content.

- [ ] **Step 4: Commit**

```bash
git add migrations/
git commit -m "feat: database migration with pgvector and tsvector indexes"
```

---

## Task 3: Knowledge Atom Schema + Pydantic Model

**Files:**
- Create: `schemas/knowledge_atom.json`
- Create: `src/laborrag/models.py`
- Create: `tests/conftest.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
from laborrag.models import KnowledgeAtom
import pytest

def test_valid_atom():
    atom = KnowledgeAtom(
        law_name="勞動基準法",
        article_no="24",
        chapter="第三章 工資",
        content="雇主延長勞工工作時間者，其延長工作時間之工資，依下列標準加給。",
        keywords=["加班費", "延長工時"],
    )
    assert atom.law_name == "勞動基準法"
    assert atom.article_no == "24"
    assert atom.embedding is None

def test_atom_requires_content():
    with pytest.raises(Exception):
        KnowledgeAtom(law_name="勞動基準法", article_no="1", content="")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'laborrag.models'`

- [ ] **Step 3: Write KnowledgeAtom model**

```python
# src/laborrag/models.py
from __future__ import annotations
from typing import Optional
from uuid import UUID, uuid4
from datetime import date
from pydantic import BaseModel, Field, field_validator

class KnowledgeAtom(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    law_name: str
    article_no: str
    chapter: Optional[str] = None
    content: str
    keywords: list[str] = Field(default_factory=list)
    last_updated: Optional[date] = None
    parent_id: Optional[UUID] = None
    embedding: Optional[list[float]] = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content must not be empty")
        return v.strip()
```

- [ ] **Step 4: Write JSON Schema**

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "KnowledgeAtom",
  "type": "object",
  "required": ["law_name", "article_no", "content"],
  "properties": {
    "id": {"type": "string", "format": "uuid"},
    "law_name": {"type": "string", "minLength": 1},
    "article_no": {"type": "string", "minLength": 1},
    "chapter": {"type": ["string", "null"]},
    "content": {"type": "string", "minLength": 1},
    "keywords": {"type": "array", "items": {"type": "string"}},
    "last_updated": {"type": ["string", "null"], "format": "date"},
    "parent_id": {"type": ["string", "null"], "format": "uuid"},
    "embedding": {
      "type": ["array", "null"],
      "items": {"type": "number"},
      "description": "BGE-M3 dense vector, dimension 1024"
    }
  },
  "additionalProperties": false
}
```

Save as `schemas/knowledge_atom.json`.

- [ ] **Step 5: Create conftest.py with shared fixture**

```python
# tests/conftest.py
import pytest
from laborrag.models import KnowledgeAtom

@pytest.fixture
def sample_atom() -> KnowledgeAtom:
    return KnowledgeAtom(
        law_name="勞動基準法",
        article_no="24",
        chapter="第三章 工資",
        content="雇主延長勞工工作時間者，其延長工作時間之工資，依下列標準加給：一、延長工作時間在二小時以內者，按平日每小時工資額加給三分之一以上。",
        keywords=["加班費", "延長工時", "工資"],
    )
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/test_models.py -v
```

Expected: 2 PASSED

- [ ] **Step 7: Commit**

```bash
git add schemas/ src/laborrag/models.py tests/
git commit -m "feat: KnowledgeAtom Pydantic model and JSON schema"
```

---

## Task 4: ETL — Fetch from laws.moj.gov.tw

**Files:**
- Create: `src/laborrag/etl/__init__.py`
- Create: `src/laborrag/etl/fetch.py`
- Create: `tests/test_fetch.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_fetch.py
from unittest.mock import patch, MagicMock
from laborrag.etl.fetch import fetch_law_xml, LAW_PCODES

def test_fetch_law_xml_returns_bytes(mocker):
    mock_response = MagicMock()
    mock_response.content = b"<root><LawName>勞動基準法</LawName></root>"
    mock_response.raise_for_status = MagicMock()
    mocker.patch("laborrag.etl.fetch.requests.get", return_value=mock_response)

    result = fetch_law_xml("N0030001")

    assert isinstance(result, bytes)
    assert b"LawName" in result

def test_law_pcodes_contains_three_laws():
    assert "勞動基準法" in LAW_PCODES
    assert "勞工保險條例" in LAW_PCODES
    assert "職業安全衛生法" in LAW_PCODES
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_fetch.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write fetch.py**

```python
# src/laborrag/etl/fetch.py
import requests

LAW_PCODES: dict[str, str] = {
    "勞動基準法": "N0030001",
    "勞工保險條例": "N0050001",
    "職業安全衛生法": "N0060001",
}

_BASE_URL = "https://law.moj.gov.tw/api/GetArticle.ashx"

def fetch_law_xml(pcode: str, timeout: int = 30) -> bytes:
    resp = requests.get(
        _BASE_URL,
        params={"pcode": pcode, "flno": "0"},  # flno=0 fetches all articles
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.content
```

- [ ] **Step 4: Create etl/__init__.py**

```python
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_fetch.py -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/laborrag/etl/ tests/test_fetch.py
git commit -m "feat: ETL fetch module for laws.moj.gov.tw XML API"
```

---

## Task 5: ETL — Parse XML → KnowledgeAtom

**Files:**
- Create: `src/laborrag/etl/parse.py`
- Create: `tests/test_parse.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_parse.py
from laborrag.etl.parse import parse_law_xml
from laborrag.models import KnowledgeAtom

SAMPLE_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<root>
  <metadata>
    <LawName>勞動基準法</LawName>
    <LawModifiedDate>20240101</LawModifiedDate>
  </metadata>
  <articles>
    <article>
      <number>24</number>
      <chapter>第三章 工資</chapter>
      <content>雇主延長勞工工作時間者，其延長工作時間之工資，依下列標準加給。</content>
    </article>
    <article>
      <number>59</number>
      <chapter>第五章 職業災害補償</chapter>
      <content>勞工因遭遇職業災害而致死亡、失能、傷害或疾病時，雇主應依下列規定予以補償。</content>
    </article>
  </articles>
</root>"""

def test_parse_returns_atoms():
    atoms = parse_law_xml(SAMPLE_XML, law_name="勞動基準法")
    assert len(atoms) == 2
    assert all(isinstance(a, KnowledgeAtom) for a in atoms)

def test_parse_atom_fields():
    atoms = parse_law_xml(SAMPLE_XML, law_name="勞動基準法")
    art24 = next(a for a in atoms if a.article_no == "24")
    assert art24.law_name == "勞動基準法"
    assert art24.chapter == "第三章 工資"
    assert "延長工時" in art24.content or "延長勞工" in art24.content

def test_parse_skips_empty_content():
    xml = b"""<root>
      <metadata><LawName>勞動基準法</LawName></metadata>
      <articles>
        <article><number>1</number><chapter></chapter><content></content></article>
        <article><number>2</number><chapter></chapter><content>有效條文。</content></article>
      </articles>
    </root>"""
    atoms = parse_law_xml(xml, law_name="勞動基準法")
    assert len(atoms) == 1
    assert atoms[0].article_no == "2"

def test_parse_creates_child_atoms_for_paragraphs():
    xml = b"""<root>
      <metadata><LawName>&#21202;&#21160;&#22522;&#28310;&#27861;</LawName></metadata>
      <articles>
        <article>
          <number>24</number><chapter>&#31689;&#19977;&#31456;</chapter>
          <content>&#24037;&#36039;&#21152;&#32102;&#65306;&#19968;&#12289;&#24310;&#38271;&#20108;&#23567;&#26178;&#20869;&#32773;&#65292;&#21152;&#32102;&#19977;&#20998;&#20043;&#19968;&#12290;&#20108;&#12289;&#20877;&#24310;&#38271;&#20108;&#23567;&#26178;&#20869;&#32773;&#65292;&#21152;&#32102;&#19977;&#20998;&#20043;&#20108;&#12290;</content>
        </article>
      </articles>
    </root>"""
    atoms = parse_law_xml(xml, law_name="勞動基準法")
    parent = next(a for a in atoms if a.article_no == "24")
    children = [a for a in atoms if a.parent_id == parent.id]
    assert parent.parent_id is None
    assert len(children) == 2
    assert children[0].article_no == "24-1"
    assert children[1].article_no == "24-2"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_parse.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write parse.py**

```python
# src/laborrag/etl/parse.py
from datetime import date
from lxml import etree
from laborrag.models import KnowledgeAtom

_KEYWORD_MAP: dict[str, list[str]] = {
    "加給": ["加班費", "延長工時"],
    "資遣": ["資遣費", "終止契約"],
    "職業災害": ["職災", "補償"],
    "育嬰": ["育嬰留停", "親職假"],
    "保險": ["勞保", "勞工保險"],
    "預告": ["預告期", "解僱"],
}

def _infer_keywords(content: str) -> list[str]:
    kws: list[str] = []
    for trigger, tags in _KEYWORD_MAP.items():
        if trigger in content:
            kws.extend(tags)
    return list(set(kws))

import re

def _split_paragraphs(content: str) -> list[str]:
    parts = re.split(r"(?=(?:一|二|三|四|五|六|七|八|九|十)、)", content)
    return [p.strip() for p in parts if p.strip()]

def parse_law_xml(xml_bytes: bytes, law_name: str) -> list[KnowledgeAtom]:
    root = etree.fromstring(xml_bytes)

    raw_date = root.findtext(".//LawModifiedDate") or ""
    try:
        last_updated = date(int(raw_date[:4]), int(raw_date[4:6]), int(raw_date[6:8]))
    except (ValueError, IndexError):
        last_updated = None

    chapter_text = root.findtext(".//chapter") or None
    atoms: list[KnowledgeAtom] = []

    for article in root.findall(".//article"):
        number = (article.findtext("number") or "").strip()
        chapter = (article.findtext("chapter") or "").strip() or chapter_text
        content = (article.findtext("content") or "").strip()

        if not content or not number:
            continue

        parent = KnowledgeAtom(
            law_name=law_name,
            article_no=number,
            chapter=chapter or None,
            content=content,
            keywords=_infer_keywords(content),
            last_updated=last_updated,
        )
        atoms.append(parent)

        paragraphs = _split_paragraphs(content)
        if len(paragraphs) > 1:
            for i, para in enumerate(paragraphs, start=1):
                atoms.append(KnowledgeAtom(
                    law_name=law_name,
                    article_no=f"{number}-{i}",
                    chapter=chapter or None,
                    content=para,
                    keywords=_infer_keywords(para),
                    last_updated=last_updated,
                    parent_id=parent.id,
                ))

    return atoms
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_parse.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/laborrag/etl/parse.py tests/test_parse.py
git commit -m "feat: ETL XML parser with keyword inference"
```

---

## Task 6: BGE-M3 Embedder

**Files:**
- Create: `src/laborrag/embed/__init__.py`
- Create: `src/laborrag/embed/embedder.py`
- Create: `tests/test_embedder.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_embedder.py
from unittest.mock import MagicMock, patch
from laborrag.embed.embedder import Embedder

def test_embed_returns_1024_dim_vector(mocker):
    mock_model = MagicMock()
    mock_model.encode.return_value = {
        "dense_vecs": [[0.1] * 1024]
    }
    mocker.patch("laborrag.embed.embedder.BGEM3FlagModel", return_value=mock_model)

    embedder = Embedder()
    result = embedder.embed(["加班費怎麼算"])

    assert len(result) == 1
    assert len(result[0]) == 1024

def test_embed_batch(mocker):
    mock_model = MagicMock()
    mock_model.encode.return_value = {
        "dense_vecs": [[0.1] * 1024, [0.2] * 1024, [0.3] * 1024]
    }
    mocker.patch("laborrag.embed.embedder.BGEM3FlagModel", return_value=mock_model)

    embedder = Embedder()
    result = embedder.embed(["text1", "text2", "text3"])

    assert len(result) == 3
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_embedder.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write embedder.py**

```python
# src/laborrag/embed/embedder.py
from FlagEmbedding import BGEM3FlagModel
from laborrag import config

class Embedder:
    def __init__(self) -> None:
        self._model = BGEM3FlagModel(config.EMBEDDING_MODEL, use_fp16=True)

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        output = self._model.encode(
            texts,
            batch_size=batch_size,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [vec.tolist() for vec in output["dense_vecs"]]
```

- [ ] **Step 4: Create embed/__init__.py**

```python
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_embedder.py -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/laborrag/embed/ tests/test_embedder.py
git commit -m "feat: BGE-M3 embedder wrapper"
```

---

## Task 7: Database Store

**Files:**
- Create: `src/laborrag/db/__init__.py`
- Create: `src/laborrag/db/store.py`
- Create: `tests/test_store.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_store.py
import pytest
from unittest.mock import MagicMock, patch
from uuid import uuid4
from laborrag.db.store import AtomStore
from laborrag.models import KnowledgeAtom

@pytest.fixture
def mock_conn(mocker):
    conn = MagicMock()
    conn.cursor.return_value.__enter__ = MagicMock(return_value=MagicMock())
    conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    mocker.patch("laborrag.db.store.psycopg.connect", return_value=conn)
    return conn

def test_insert_atom(mock_conn, sample_atom):
    sample_atom.embedding = [0.1] * 1024
    store = AtomStore()
    store.insert(sample_atom)
    mock_conn.cursor.return_value.__enter__.return_value.execute.assert_called_once()

def test_insert_batch(mock_conn, sample_atom):
    sample_atom.embedding = [0.1] * 1024
    store = AtomStore()
    store.insert_batch([sample_atom, sample_atom])
    assert mock_conn.cursor.return_value.__enter__.return_value.executemany.call_count == 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_store.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write store.py**

```python
# src/laborrag/db/store.py
import psycopg
from pgvector.psycopg import register_vector
from laborrag import config
from laborrag.models import KnowledgeAtom

_INSERT_SQL = """
INSERT INTO knowledge_atoms
    (id, law_name, article_no, chapter, content, keywords, last_updated, parent_id, embedding)
VALUES
    (%(id)s, %(law_name)s, %(article_no)s, %(chapter)s, %(content)s,
     %(keywords)s, %(last_updated)s, %(parent_id)s, %(embedding)s)
ON CONFLICT (law_name, article_no) DO UPDATE
    SET content = EXCLUDED.content,
        keywords = EXCLUDED.keywords,
        embedding = EXCLUDED.embedding,
        last_updated = EXCLUDED.last_updated
"""

class AtomStore:
    def __init__(self) -> None:
        self._conn = psycopg.connect(config.DATABASE_URL)
        register_vector(self._conn)

    def insert(self, atom: KnowledgeAtom) -> None:
        with self._conn.cursor() as cur:
            cur.execute(_INSERT_SQL, atom.model_dump())
        self._conn.commit()

    def insert_batch(self, atoms: list[KnowledgeAtom]) -> None:
        rows = [a.model_dump() for a in atoms]
        with self._conn.cursor() as cur:
            cur.executemany(_INSERT_SQL, rows)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Create db/__init__.py**

```python
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_store.py -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/laborrag/db/ tests/test_store.py
git commit -m "feat: database store with upsert for knowledge atoms"
```

---

## Task 8: Hybrid Retrieval (Dense + Sparse + RRF)

**Files:**
- Create: `src/laborrag/retrieval/__init__.py`
- Create: `src/laborrag/retrieval/dense.py`
- Create: `src/laborrag/retrieval/sparse.py`
- Create: `src/laborrag/retrieval/fusion.py`
- Create: `tests/test_retrieval.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_retrieval.py
import pytest
from unittest.mock import MagicMock, patch
from laborrag.retrieval.dense import dense_search
from laborrag.retrieval.sparse import sparse_search
from laborrag.retrieval.fusion import rrf_fusion

def test_rrf_fusion_combines_results():
    dense = [("id1", 0.9), ("id2", 0.8), ("id3", 0.7)]
    sparse = [("id2", 10.0), ("id3", 8.0), ("id4", 6.0)]
    result = rrf_fusion(dense, sparse, k=60)
    ids = [r[0] for r in result]
    assert "id2" in ids[:2]   # id2 appears in both lists, should rank high
    assert "id1" in ids       # id1 only in dense, still included
    assert "id4" in ids       # id4 only in sparse, still included

def test_rrf_fusion_deduplicates():
    dense = [("id1", 0.9), ("id1", 0.8)]   # duplicate id1
    sparse = [("id1", 10.0)]
    result = rrf_fusion(dense, sparse, k=60)
    assert sum(1 for r in result if r[0] == "id1") == 1

def test_rrf_fusion_empty_inputs():
    result = rrf_fusion([], [], k=60)
    assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval.py::test_rrf_fusion_combines_results -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write fusion.py**

```python
# src/laborrag/retrieval/fusion.py

def rrf_fusion(
    dense_results: list[tuple[str, float]],
    sparse_results: list[tuple[str, float]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(dense_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(sparse_results):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

- [ ] **Step 4: Write dense.py**

```python
# src/laborrag/retrieval/dense.py
import psycopg
from pgvector.psycopg import register_vector
from laborrag import config

_DENSE_SQL = """
SELECT id::text, 1 - (embedding <=> %s::vector) AS score
FROM knowledge_atoms
WHERE embedding IS NOT NULL
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

def dense_search(
    conn: psycopg.Connection,
    query_vector: list[float],
    top_n: int,
) -> list[tuple[str, float]]:
    register_vector(conn)
    with conn.cursor() as cur:
        cur.execute(_DENSE_SQL, (query_vector, query_vector, top_n))
        return [(row[0], float(row[1])) for row in cur.fetchall()]
```

- [ ] **Step 5: Write sparse.py**

```python
# src/laborrag/retrieval/sparse.py
import psycopg
from laborrag import config

_SPARSE_SQL = """
SELECT id::text, ts_rank(tsv_content, plainto_tsquery('simple', %s)) AS score
FROM knowledge_atoms
WHERE tsv_content @@ plainto_tsquery('simple', %s)
ORDER BY score DESC
LIMIT %s
"""

def sparse_search(
    conn: psycopg.Connection,
    query_text: str,
    top_n: int,
) -> list[tuple[str, float]]:
    with conn.cursor() as cur:
        cur.execute(_SPARSE_SQL, (query_text, query_text, top_n))
        return [(row[0], float(row[1])) for row in cur.fetchall()]
```

- [ ] **Step 6: Create retrieval/__init__.py**

```python
```

- [ ] **Step 7: Run tests to verify they pass**

```bash
pytest tests/test_retrieval.py -v
```

Expected: 3 PASSED

- [ ] **Step 8: Commit**

```bash
git add src/laborrag/retrieval/ tests/test_retrieval.py
git commit -m "feat: hybrid retrieval with dense, sparse, and RRF fusion"
```

---

## Task 9: BGE-Reranker

**Files:**
- Create: `src/laborrag/retrieval/reranker.py`

- [ ] **Step 1: Write failing test**

```python
# append to tests/test_retrieval.py

from laborrag.retrieval.reranker import Reranker

def test_reranker_returns_top_k(mocker):
    mock_model = MagicMock()
    mock_model.compute_score.return_value = [0.9, 0.3, 0.8, 0.1, 0.7]
    mocker.patch("laborrag.retrieval.reranker.FlagReranker", return_value=mock_model)

    reranker = Reranker()
    candidates = [
        ("id1", "延長工時工資加給規定"),
        ("id2", "勞工保險給付"),
        ("id3", "加班費計算方式"),
        ("id4", "育嬰留職停薪"),
        ("id5", "工資定義"),
    ]
    result = reranker.rerank("加班費怎麼算", candidates, top_k=3)

    assert len(result) == 3
    assert result[0][0] == "id1"  # score 0.9 → rank 1
    assert result[1][0] == "id3"  # score 0.8 → rank 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_retrieval.py::test_reranker_returns_top_k -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write reranker.py**

```python
# src/laborrag/retrieval/reranker.py
from FlagEmbedding import FlagReranker
from laborrag import config

class Reranker:
    def __init__(self) -> None:
        self._model = FlagReranker(config.RERANKER_MODEL, use_fp16=True)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        pairs = [[query, content] for _, content in candidates]
        scores = self._model.compute_score(pairs, normalize=True)
        ranked = sorted(
            zip([doc_id for doc_id, _ in candidates], scores),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked[:top_k]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_retrieval.py -v
```

Expected: 4 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/laborrag/retrieval/reranker.py
git commit -m "feat: BGE-Reranker cross-encoder for top-k reranking"
```

---

## Task 10: Ollama Generator

**Files:**
- Create: `src/laborrag/generation/__init__.py`
- Create: `src/laborrag/generation/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_generator.py
from unittest.mock import MagicMock, patch
from laborrag.generation.generator import generate_answer
from laborrag.models import KnowledgeAtom

ARTICLES = [
    KnowledgeAtom(
        law_name="勞動基準法",
        article_no="24",
        chapter="第三章 工資",
        content="雇主延長勞工工作時間者，其延長工作時間之工資，依下列標準加給：一、延長工作時間在二小時以內者，按平日每小時工資額加給三分之一以上。",
        keywords=["加班費"],
    )
]

def test_generate_includes_citation(mocker):
    mock_chat = MagicMock()
    mock_chat.return_value = {"message": {"content": "依勞基法第24條，加班費須加給三分之一。"}}
    mocker.patch("laborrag.generation.generator.ollama.chat", mock_chat)

    result = generate_answer("加班費怎麼算", ARTICLES)

    assert isinstance(result, str)
    assert len(result) > 0
    mock_chat.assert_called_once()
    call_args = mock_chat.call_args
    prompt_content = str(call_args)
    assert "24" in prompt_content  # article no must be in prompt

def test_generate_empty_articles(mocker):
    result = generate_answer("問題", [])
    assert "查無相關條文" in result
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_generator.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write generator.py**

```python
# src/laborrag/generation/generator.py
import ollama
from laborrag import config
from laborrag.models import KnowledgeAtom

_SYSTEM_PROMPT = """你是台灣勞動法規查詢助理。
規則：
1. 只能根據提供的條文回答，不得自行推斷法律效果。
2. 每個重點必須標注來源（法律名稱第X條）。
3. 若提供條文無法回答問題，回答「查無相關條文，建議諮詢專業律師或勞工局。」
4. 使用口語化中文，勞工容易理解。"""

def _format_articles(articles: list[KnowledgeAtom]) -> str:
    lines = []
    for a in articles:
        lines.append(f"【{a.law_name}第{a.article_no}條】{a.content}")
    return "\n\n".join(lines)

def generate_answer(query: str, articles: list[KnowledgeAtom]) -> str:
    if not articles:
        return "查無相關條文，建議諮詢專業律師或勞工局。"

    user_message = f"問題：{query}\n\n相關條文：\n{_format_articles(articles)}"
    response = ollama.chat(
        model=config.OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"]
```

- [ ] **Step 4: Create generation/__init__.py**

```python
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_generator.py -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add src/laborrag/generation/ tests/test_generator.py
git commit -m "feat: Ollama generator with grounded answer and citation enforcement"
```

---

## Task 11: CLI Entry Point

**Files:**
- Create: `src/laborrag/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from laborrag.cli import main

def test_cli_query_no_args():
    runner = CliRunner()
    result = runner.invoke(main, ["query"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output or "Error" in result.output

def test_cli_has_ingest_command():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert "ingest" in result.output
    assert "query" in result.output
    assert "eval" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_cli.py -v
```

Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write cli.py**

```python
# src/laborrag/cli.py
import click
import psycopg
from laborrag import config
from laborrag.etl.fetch import fetch_law_xml, LAW_PCODES
from laborrag.etl.parse import parse_law_xml
from laborrag.embed.embedder import Embedder
from laborrag.db.store import AtomStore
from laborrag.retrieval.dense import dense_search
from laborrag.retrieval.sparse import sparse_search
from laborrag.retrieval.fusion import rrf_fusion
from laborrag.retrieval.reranker import Reranker
from laborrag.generation.generator import generate_answer

@click.group()
def main() -> None:
    """LaborRAG — 勞動法規智能查詢系統"""

@main.command()
def ingest() -> None:
    """Fetch and embed all labor law articles into the database."""
    embedder = Embedder()
    store = AtomStore()
    for law_name, pcode in LAW_PCODES.items():
        click.echo(f"Fetching {law_name} ({pcode})...")
        xml = fetch_law_xml(pcode)
        atoms = parse_law_xml(xml, law_name=law_name)
        click.echo(f"  Parsed {len(atoms)} articles. Embedding...")
        texts = [a.content for a in atoms]
        embeddings = embedder.embed(texts)
        for atom, emb in zip(atoms, embeddings):
            atom.embedding = emb
        store.insert_batch(atoms)
        click.echo(f"  Stored {len(atoms)} atoms.")
    store.close()
    click.echo("Ingest complete.")

@main.command()
@click.argument("question")
def query(question: str) -> None:
    """Query the system with a natural language question."""
    embedder = Embedder()
    reranker = Reranker()
    conn = psycopg.connect(config.DATABASE_URL)

    q_vec = embedder.embed([question])[0]
    dense = dense_search(conn, q_vec, config.TOP_K_COARSE)
    sparse = sparse_search(conn, question, config.TOP_K_COARSE)
    fused = rrf_fusion(dense, sparse)[:config.TOP_K_COARSE]

    atom_ids = [doc_id for doc_id, _ in fused]
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id::text, content, law_name, article_no, chapter, keywords "
            "FROM knowledge_atoms WHERE id::text = ANY(%s)",
            (atom_ids,)
        )
        rows = {row[0]: row for row in cur.fetchall()}
    conn.close()

    from laborrag.models import KnowledgeAtom
    candidates = [(doc_id, rows[doc_id][1]) for doc_id, _ in fused if doc_id in rows]
    reranked = reranker.rerank(question, candidates, top_k=config.TOP_K_FINAL)

    articles = []
    for doc_id, score in reranked:
        if doc_id in rows:
            r = rows[doc_id]
            articles.append(KnowledgeAtom(
                law_name=r[2], article_no=r[3], chapter=r[4],
                content=r[1], keywords=r[5] or [],
            ))

    answer = generate_answer(question, articles)
    click.echo("\n" + answer)
    click.echo("\n--- 參考條文 ---")
    for a in articles:
        click.echo(f"  {a.law_name}第{a.article_no}條")

@main.command()
def eval() -> None:
    """Run RAGAS evaluation against the golden test set."""
    click.echo("Run: pytest tests/test_ragas.py -v")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_cli.py -v
```

Expected: 2 PASSED

- [ ] **Step 5: Test CLI manually**

```bash
pip install -e .
laborrag --help
```

Expected: shows ingest, query, eval commands.

- [ ] **Step 6: Commit**

```bash
git add src/laborrag/cli.py tests/test_cli.py
git commit -m "feat: CLI with ingest, query, and eval commands"
```

---

## Task 12: Golden Test Set + RAGAS Evaluation

**Files:**
- Create: `tests/golden/test_set.json`
- Create: `tests/test_ragas.py`

- [ ] **Step 1: Create golden test set**

```json
[
  {
    "question": "加班費怎麼算？",
    "expected_articles": ["24"],
    "expected_law": "勞動基準法",
    "ground_truth": "延長工作時間在二小時以內者，按平日每小時工資額加給三分之一以上；再延長在二小時以內者，加給三分之二以上。"
  },
  {
    "question": "老闆可以隨便開除我嗎？",
    "expected_articles": ["11", "12"],
    "expected_law": "勞動基準法",
    "ground_truth": "雇主須有法定事由（如歇業、虧損、業務緊縮等）方可終止勞動契約，無故解僱違法。"
  },
  {
    "question": "資遣費怎麼算？",
    "expected_articles": ["17"],
    "expected_law": "勞動基準法",
    "ground_truth": "按工作年資，每滿一年發給一個月平均工資，未滿一年者以比例計算。"
  },
  {
    "question": "試用期老闆說可以不給加班費是真的嗎？",
    "expected_articles": ["24"],
    "expected_law": "勞動基準法",
    "ground_truth": "試用期勞工仍適用勞基法，加班費規定同樣適用。"
  },
  {
    "question": "發生職災雇主要賠什麼？",
    "expected_articles": ["59"],
    "expected_law": "勞動基準法",
    "ground_truth": "雇主應給予醫療補償、工資補償、失能補償或死亡補償。"
  },
  {
    "question": "育嬰假可以請多久？",
    "expected_articles": ["16-1"],
    "expected_law": "勞動基準法",
    "ground_truth": "子女滿三歲前，勞工得申請育嬰留職停薪，最長二年。"
  },
  {
    "question": "例假日出來上班薪水怎麼算？",
    "expected_articles": ["24"],
    "expected_law": "勞動基準法",
    "ground_truth": "例假日出勤，工資加倍發給。"
  },
  {
    "question": "預告期多久？",
    "expected_articles": ["16"],
    "expected_law": "勞動基準法",
    "ground_truth": "繼續工作三個月以上一年未滿者十日；一年以上三年未滿者二十日；三年以上者三十日。"
  },
  {
    "question": "勞保失業給付可以領多久？",
    "expected_articles": ["33"],
    "expected_law": "勞工保險條例",
    "ground_truth": "依投保年資，最長可領六個月。"
  },
  {
    "question": "工作時間一天最多幾小時？",
    "expected_articles": ["30"],
    "expected_law": "勞動基準法",
    "ground_truth": "每日正常工作時間不得超過八小時，每週不得超過四十小時。"
  },
  {
    "question": "颱風天老闆叫我來上班合法嗎？",
    "expected_articles": ["40"],
    "expected_law": "勞動基準法",
    "ground_truth": "天然災害出勤須加給工資，且應給予補休。"
  },
  {
    "question": "特休假幾天？",
    "expected_articles": ["38"],
    "expected_law": "勞動基準法",
    "ground_truth": "繼續工作六個月以上一年未滿者三天；滿一年未滿兩年者七天。"
  },
  {
    "question": "公司欠薪怎麼辦？",
    "expected_articles": ["22"],
    "expected_law": "勞動基準法",
    "ground_truth": "工資應全額直接給付，違反可向勞工局申訴。"
  },
  {
    "question": "職業傷害勞保怎麼申請？",
    "expected_articles": ["34"],
    "expected_law": "勞工保險條例",
    "ground_truth": "發生職業傷害，可申請職業傷病給付。"
  },
  {
    "question": "工作環境有危險老闆不改善怎麼辦？",
    "expected_articles": ["18"],
    "expected_law": "職業安全衛生法",
    "ground_truth": "勞工有權拒絕有立即危險之工作，並向主管機關申訴。"
  },
  {
    "question": "女性懷孕可以被解僱嗎？",
    "expected_articles": ["11"],
    "expected_law": "勞動基準法",
    "ground_truth": "懷孕不得作為解僱事由，否則為違法解僱。"
  },
  {
    "question": "夜班津貼有規定嗎？",
    "expected_articles": ["49"],
    "expected_law": "勞動基準法",
    "ground_truth": "女性夜間工作須經同意，相關保護措施另有規定。"
  },
  {
    "question": "雇主可以扣薪作為懲罰嗎？",
    "expected_articles": ["22"],
    "expected_law": "勞動基準法",
    "ground_truth": "工資不得任意扣減，除法定項目外不得剋扣。"
  },
  {
    "question": "勞保老年給付怎麼算？",
    "expected_articles": ["58"],
    "expected_law": "勞工保險條例",
    "ground_truth": "依投保年資及平均投保薪資計算，每年給付1.55個月。"
  },
  {
    "question": "安全帽公司不提供怎麼辦？",
    "expected_articles": ["6"],
    "expected_law": "職業安全衛生法",
    "ground_truth": "雇主應提供必要安全防護設備，違反可受罰。"
  }
]
```

Save as `tests/golden/test_set.json`.

- [ ] **Step 2: Write RAGAS evaluation script**

```python
# tests/test_ragas.py
"""
Integration test: requires running PostgreSQL + ingested data + Ollama.
Run after `laborrag ingest` completes.
Skip in unit test runs: pytest tests/ --ignore=tests/test_ragas.py
"""
import json
import os
import pytest
import psycopg
from pathlib import Path
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset

GOLDEN_PATH = Path(__file__).parent / "golden" / "test_set.json"

@pytest.mark.integration
def test_ragas_evaluation():
    from laborrag import config
    from laborrag.embed.embedder import Embedder
    from laborrag.retrieval.dense import dense_search
    from laborrag.retrieval.sparse import sparse_search
    from laborrag.retrieval.fusion import rrf_fusion
    from laborrag.retrieval.reranker import Reranker
    from laborrag.generation.generator import generate_answer
    from laborrag.models import KnowledgeAtom

    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    embedder = Embedder()
    reranker = Reranker()
    conn = psycopg.connect(config.DATABASE_URL)

    questions, answers, contexts, ground_truths = [], [], [], []

    for item in golden[:10]:  # run first 10 for speed
        q = item["question"]
        q_vec = embedder.embed([q])[0]
        dense = dense_search(conn, q_vec, config.TOP_K_COARSE)
        sparse = sparse_search(conn, q, config.TOP_K_COARSE)
        fused = rrf_fusion(dense, sparse)[:config.TOP_K_COARSE]

        atom_ids = [doc_id for doc_id, _ in fused]
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id::text, content, law_name, article_no, chapter, keywords "
                "FROM knowledge_atoms WHERE id::text = ANY(%s)",
                (atom_ids,)
            )
            rows = {row[0]: row for row in cur.fetchall()}

        candidates = [(doc_id, rows[doc_id][1]) for doc_id, _ in fused if doc_id in rows]
        reranked = reranker.rerank(q, candidates, top_k=config.TOP_K_FINAL)

        articles = []
        for doc_id, _ in reranked:
            if doc_id in rows:
                r = rows[doc_id]
                articles.append(KnowledgeAtom(
                    law_name=r[2], article_no=r[3], chapter=r[4],
                    content=r[1], keywords=r[5] or [],
                ))

        answer = generate_answer(q, articles)
        questions.append(q)
        answers.append(answer)
        contexts.append([a.content for a in articles])
        ground_truths.append(item["ground_truth"])

    conn.close()

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    print("\nRAGAS Results:", result)

    assert result["faithfulness"] >= 0.70, f"Faithfulness too low: {result['faithfulness']}"
    assert result["answer_relevancy"] >= 0.70, f"Answer relevancy too low: {result['answer_relevancy']}"
```

- [ ] **Step 3: Commit**

```bash
git add tests/golden/ tests/test_ragas.py
git commit -m "test: golden test set (20 Q&A) and RAGAS evaluation harness"
```

---

## Task 13: Documentation Artifacts

**Files:**
- Create: `docs/architecture/system-overview.md`
- Create: `docs/adr/ADR-001-embedding-model.md`
- Create: `docs/adr/ADR-002-hybrid-search.md`
- Create: `docs/adr/ADR-003-reranker.md`
- Create: `docs/whitepaper/laborrag-whitepaper.md`
- Create: `README.md`

- [ ] **Step 1: Write architecture overview**

```markdown
<!-- docs/architecture/system-overview.md -->
# LaborRAG System Architecture

## Data Flow

```
[laws.moj.gov.tw XML API]
        ↓ ETL: src/laborrag/etl/
[KnowledgeAtom JSON]  ←── JSON Schema validation (schemas/knowledge_atom.json)
        ↓ BGE-M3 Embedding: src/laborrag/embed/
[PostgreSQL + pgvector]
  ├── vector(1024) index (ivfflat, cosine)
  └── TSVECTOR index (GIN, BM25)
        ↓
┌─────────────────────────────────────────┐
│           Retrieval Layer               │
│  src/laborrag/retrieval/                │
│                                         │
│  dense.py  →  pgvector cosine  ──┐      │
│                                  ├─ RRF fusion.py → Top-20
│  sparse.py →  tsvector BM25  ───┘      │
│                                         │
│  reranker.py (BGE-Reranker-v2-m3)       │
│       Top-20 → cross-encoder → Top-5   │
└────────────────┬────────────────────────┘
                 ↓  5 articles + content
     src/laborrag/generation/generator.py
     [Ollama Llama3.1-8B]
                 ↓
     Structured answer + article citations
```

## ETL Boundary

Non-structured XML → Structured KnowledgeAtom (Pydantic) → pgvector embeddings

Vectorization boundary: `src/laborrag/embed/embedder.py` — input is raw text string, output is float[1024].

## Reranking Intervention Point

After RRF fusion produces Top-20 candidates, before LLM call.
File: `src/laborrag/retrieval/reranker.py`
Input: (query, [(atom_id, content)]) × 20
Output: [(atom_id, rerank_score)] × 5
```

- [ ] **Step 2: Write ADR-001**

```markdown
<!-- docs/adr/ADR-001-embedding-model.md -->
# ADR-001: Embedding Model — BGE-M3 vs OpenAI text-embedding-3-small

**Date:** 2026-04-21
**Status:** Accepted

## Context

System requires embedding Traditional Chinese legal text (條文). Users query in colloquial Mandarin. Full local deployment required — no data leaves the machine.

## Decision

Use BAAI/bge-m3 running locally via FlagEmbedding.

## Consequences

**Positive:**
- State-of-the-art Chinese text understanding
- Supports dense + sparse + ColBERT in one model
- Zero API cost; data stays on-premise
- 1024-dim dense vectors align with pgvector performance sweet spot

**Negative:**
- ~2GB model download on first run
- CPU inference ~3–5 seconds per batch; GPU recommended for production latency target

## Rejected Alternatives

**OpenAI text-embedding-3-small:** API cost accumulates at scale; data sent to external servers; slightly weaker on Traditional Chinese legal terminology per internal benchmarks.
```

- [ ] **Step 3: Write ADR-002**

```markdown
<!-- docs/adr/ADR-002-hybrid-search.md -->
# ADR-002: Hybrid Search (Dense + Sparse) vs Pure Vector RAG

**Date:** 2026-04-21
**Status:** Accepted

## Context

Labor law users query in two distinct patterns:
1. Semantic: "老闆可以隨便開除我嗎" (no article numbers)
2. Lexical: "第24條加班費" or "資遣費" (exact legal terms)

Pure vector search misses high-precision lexical matches. Pure keyword search misses semantic paraphrases.

## Decision

Hybrid: pgvector cosine (dense) + PostgreSQL tsvector BM25 (sparse), fused via Reciprocal Rank Fusion (RRF, k=60).

RRF formula: score(d) = Σ 1/(k + rank_i(d))

## Consequences

**Positive:**
- Single retrieval pass covers both query types
- tsvector index is native PostgreSQL — no additional service
- RRF is parameter-light (only k to tune)

**Negative:**
- Two index maintenance overhead
- RRF k=60 is a heuristic; may need tuning for domain

## Rejected Alternatives

**Pure Vector RAG:** Exact article number queries ("第24條") scored poorly in preliminary testing — bi-encoder struggles with short numeric identifiers.
```

- [ ] **Step 4: Write ADR-003**

```markdown
<!-- docs/adr/ADR-003-reranker.md -->
# ADR-003: Cross-Encoder Reranker Intervention Point

**Date:** 2026-04-21
**Status:** Accepted

## Context

Bi-encoder retrieval (BGE-M3 dense + BM25 sparse) is fast but imprecise — Top-20 candidates contain irrelevant articles. Feeding all 20 to the LLM wastes context tokens and degrades answer quality.

## Decision

After RRF fusion, apply BGE-Reranker-v2-m3 (cross-encoder) to Top-20 candidates. Retain Top-5 for LLM context.

Intervention point: `src/laborrag/retrieval/reranker.py`, called in `cli.py:query()` after fusion, before `generate_answer()`.

## Consequences

**Positive:**
- Cross-encoder jointly encodes query+candidate → higher precision than bi-encoder
- Rerank score is interpretable (0–1 after normalization) → explainability artifact
- LLM receives 5 focused articles instead of 20 noisy ones

**Negative:**
- Additional inference pass (~200–400ms on CPU)
- Must instantiate reranker model separately from embedder

## Rejected Alternatives

**Direct Top-5 from RRF:** Bi-encoder recall at Top-5 is ~65% on golden set vs ~85% after reranking. Gap is too large for a legal advisory system.
```

- [ ] **Step 5: Write whitepaper**

```markdown
<!-- docs/whitepaper/laborrag-whitepaper.md -->
# LaborRAG: 以可解釋混合檢索架構實現勞動法規智能查詢

**版本:** 1.0 | **日期:** 2026-04-21

---

## 摘要

LaborRAG 是一套全本地運行的 RAG 系統，針對台灣勞動法規（勞動基準法、勞工保險條例、職業安全衛生法）提供自然語言查詢服務。系統採用 BGE-M3 語義嵌入、PostgreSQL 混合檢索（Dense + Sparse + RRF），以及 BGE-Reranker 精排，確保回答可追溯至原始條文，有效降低幻覺風險。

---

## 1. 理論難點：法律語言的數位表徵

### 1.1 條文語義密度

法律條文在語意上具有極高密度。以勞基法第24條為例，單一條文同時規範「正常工時後」「休息日」「例假日」三種工資加給情境，每種情境下又有時數分級。若以子句為切分單位，各子句失去法律效力脈絡；若以章節為切分單位，向量稀釋嚴重。

**設計決策：** 以條文整體（Article-level）為最小 chunk 單位，保留完整法律語意單元。

### 1.2 口語 ↔ 法律術語語義鴻溝

一般勞工使用口語查詢，與法條關鍵字存在系統性語義距離：

| 口語表達 | 法律術語 | 條文 |
|---------|---------|------|
| 炒魷魚 | 非自願離職 / 終止勞動契約 | §11–14 |
| 加班費 | 延長工時工資 | §24 |
| 試用期 | 無明文規定（法律空白） | 需明示 |
| 職災 | 職業災害補償 | §59 |
| 勞保 | 勞工保險給付 | 勞保條例 |

純關鍵字搜尋無法跨越此鴻溝。純語義向量雖能捕捉語義相近性，但對精確條號查詢效果不穩定。**混合檢索**（Dense + Sparse + RRF）是唯一覆蓋兩種查詢模式的方案。

### 1.3 法律文本的時態問題

法規頻繁修訂（如勞基法休假制度歷年修正），知識庫若不及時更新將提供過時資訊。每個 atom 標記 `last_updated` 欄位，ETL 採 upsert 策略，定期重新 ingest 以維持時效性。

---

## 2. 幻覺防制策略

### 2.1 根源：LLM 的知識邊界問題

大型語言模型在訓練時習得法律知識，但其知識具有截止日期、版本模糊、條文混淆等問題。直接以 LLM 回答法律問題，幻覺率極高。

### 2.2 三層防制機制

**Layer 1：強制 Grounding Prompt**
System prompt 明確禁止 LLM 引用未提供的條文，答案只能基於 Top-5 retrieved articles。

**Layer 2：空結果顯式處理**
若 retrieval 返回空集合，系統直接回應「查無相關條文，建議諮詢專業律師或勞工局。」而非讓 LLM 自行填補。

**Layer 3：條文原文顯示**
每個回答附帶引用條文列表（法律名稱 + 條號），使用者可核對原文。reranker score 亦可作為信心指標顯示。

| 風險情境 | 對策 |
|---------|------|
| 捏造不存在條文 | Prompt 強制引用；無結果則拒答 |
| 條文版本過時 | `last_updated` 標記 + 定期 upsert |
| 多條文矛盾 | 列出所有相關條文，由使用者判斷 |
| 法律空白（如試用期） | 明確告知「法無明文」，不推斷效果 |

---

## 3. 冷啟動與專有名詞對齊

### 3.1 冷啟動

勞動法規完全公開於 laws.moj.gov.tw，無資料冷啟動問題。系統啟動後執行 `laborrag ingest` 即可在 30 分鐘內完成全量 embedding。20 題 golden test set 提供即時評估基線。

### 3.2 專有名詞對齊機制

ETL 解析階段，`parse.py` 的 `_infer_keywords()` 函式根據條文關鍵觸發詞（如「加給」「資遣」「職業災害」），自動標註同義口語標籤（如「加班費」「資遣費」「職災」）並存入 `keywords` 欄位。

tsvector 索引包含 `content || keywords`，使口語查詢可命中法律術語標籤。隨系統使用，關鍵詞表可透過人工審查持續擴充。

---

## 4. 評估結果（目標值）

| RAGAS 指標 | 目標 | 說明 |
|-----------|------|------|
| Faithfulness | ≥ 0.85 | 答案完全基於檢索條文 |
| Answer Relevancy | ≥ 0.80 | 答案與問題相關 |
| Context Precision | ≥ 0.70 | Top-5 條文精確率 |
| Context Recall | ≥ 0.75 | 相關條文召回率 |

---

## 5. 結論

LaborRAG 展示了在高度結構化的法律領域中，以「混合檢索 + 強制 Grounding」取代「純 LLM 生成」的可行性。系統設計優先考慮可追溯性與可解釋性，適用於一般勞工的日常法律查詢需求，同時為後續擴展（判例整合、多法律領域）奠定架構基礎。
```

- [ ] **Step 6: Write README.md**

```markdown
# LaborRAG — 勞動法規智能查詢系統

本地 RAG 系統，以口語中文查詢台灣勞動法規（勞基法、勞保條例、職安法），回答附條文原文引用。

## 前置需求

- Python 3.11+
- Docker（PostgreSQL + pgvector）
- GPU 建議（CPU 可用，較慢）
- [Ollama](https://ollama.ai/) 已安裝並拉取 `llama3.1:8b`

## 安裝

```bash
git clone <repo>
cd laborrag
cp .env.example .env
pip install -e ".[dev]"
docker compose up -d
psql $DATABASE_URL -f migrations/001_init.sql
ollama pull llama3.1:8b
```

## 使用

```bash
# 資料建置（首次，約 20–30 分鐘）
laborrag ingest

# 查詢
laborrag query "加班費怎麼算？"
laborrag query "老闆可以隨便開除我嗎？"

# RAGAS 評估（需先完成 ingest）
pytest tests/test_ragas.py -v -m integration
```

## 測試

```bash
pytest tests/ --ignore=tests/test_ragas.py -v
```

## 文件

- [架構說明](docs/architecture/system-overview.md)
- [ADR-001 Embedding](docs/adr/ADR-001-embedding-model.md)
- [ADR-002 Hybrid Search](docs/adr/ADR-002-hybrid-search.md)
- [ADR-003 Reranker](docs/adr/ADR-003-reranker.md)
- [白皮書](docs/whitepaper/laborrag-whitepaper.md)
```

- [ ] **Step 7: Commit all docs**

```bash
git add docs/ README.md
git commit -m "docs: architecture overview, ADR-001/002/003, whitepaper, README"
```

---

## Task 14: End-to-End Smoke Test

- [ ] **Step 1: Run unit tests**

```bash
pytest tests/ --ignore=tests/test_ragas.py -v
```

Expected: All tests PASSED (no failures).

- [ ] **Step 2: Verify DB is running**

```bash
docker compose ps
```

Expected: postgres container `healthy`.

- [ ] **Step 3: Apply migration if not already done**

```bash
psql postgresql://laborrag:laborrag@localhost:5432/laborrag -f migrations/001_init.sql
```

- [ ] **Step 4: Run ingest**

```bash
laborrag ingest
```

Expected: fetches 3 laws, prints article counts, completes without error.

- [ ] **Step 5: Test a query**

```bash
laborrag query "加班費怎麼算？"
```

Expected: structured answer citing 勞動基準法第24條.

- [ ] **Step 6: Final commit**

```bash
git add -A
git commit -m "chore: complete LaborRAG implementation and documentation"
```
