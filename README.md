# LaborRAG — 勞動法規智能查詢系統

> 以口語中文查詢台灣勞動法規，回答附條文原文引用，全程本地運行，零幻覺設計。

---

## 系統架構

```
[laws.moj.gov.tw XML API]
        ↓ ETL
[Knowledge Atoms]  →  BGE-M3 Embedding
        ↓
[PostgreSQL + pgvector]
        ↓
Dense Search (pgvector) ─┐
                          ├─ RRF Fusion → BGE-Reranker → Top-5
Sparse Search (BM25)  ───┘
        ↓
[Ollama Llama3.1-8B]
        ↓
回答 + 條文引用
```

## 技術選型

| 元件 | 選型 | 理由 |
|------|------|------|
| Embedding | BGE-M3 (BAAI/bge-m3) | 本地、中文強、1024-dim |
| Reranker | BGE-Reranker-v2-m3 | Cross-encoder 精排 |
| LLM | Llama3.1-8B (Ollama) | 本地、免費 |
| Vector DB | PostgreSQL 16 + pgvector | Hybrid search 原生支援 |
| Retrieval | Dense + Sparse + RRF | 覆蓋語義與關鍵字查詢 |
| 評估框架 | RAGAS | Faithfulness / Relevancy |
| 資料來源 | laws.moj.gov.tw XML API | 官方公開法規資料 |

## 前置需求

- Python 3.11+
- Docker（用於 PostgreSQL + pgvector）
- [Ollama](https://ollama.ai/)
- GPU 建議（CPU 可用，速度較慢）

## 安裝與啟動

```bash
git clone https://github.com/<USERNAME>/laborrag.git
cd laborrag

# 環境設定
cp .env.example .env

# 安裝 Python 套件
pip install -e ".[dev]"

# 啟動資料庫
docker compose up -d

# 建立資料表
psql postgresql://laborrag:laborrag@localhost:5432/laborrag -f migrations/001_init.sql

# 下載 LLM
ollama pull llama3.1:8b
```

## 使用方式

```bash
# 第一步：建置知識庫（首次約 20–30 分鐘）
laborrag ingest

# 查詢
laborrag query "加班費怎麼算？"
laborrag query "老闆可以隨便開除我嗎？"
laborrag query "資遣費怎麼計算？"

# RAGAS 評估（需先完成 ingest）
pytest tests/test_ragas.py -v -m integration
```

## 執行單元測試

```bash
pytest tests/ --ignore=tests/test_ragas.py -v
```

## 專案文件

| 文件 | 說明 |
|------|------|
| [架構說明](docs/architecture/system-overview.md) | 系統架構圖、ETL 邊界、Reranking 介入點 |
| [ADR-001 Embedding Model](docs/adr/ADR-001-embedding-model.md) | BGE-M3 vs OpenAI 決策記錄 |
| [ADR-002 Hybrid Search](docs/adr/ADR-002-hybrid-search.md) | 混合檢索 vs 純向量 RAG |
| [ADR-003 Reranker](docs/adr/ADR-003-reranker.md) | Cross-encoder 介入點設計 |
| [白皮書](docs/whitepaper/laborrag-whitepaper.md) | 理論難點、幻覺防制、專有名詞對齊 |

## 評估目標（RAGAS）

| 指標 | 目標 |
|------|------|
| Faithfulness | ≥ 0.85 |
| Answer Relevancy | ≥ 0.80 |
| Context Precision | ≥ 0.70 |
| Context Recall | ≥ 0.75 |

## 涵蓋法規

- 勞動基準法（含施行細則）
- 勞工保險條例
- 職業安全衛生法
