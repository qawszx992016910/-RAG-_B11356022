# LaborRAG — 勞動法規智能查詢系統 設計規格

**Date:** 2026-04-21  
**Status:** Approved  
**Domain:** 勞動法規（勞基法、勞保條例、職安法）  
**Target User:** 一般勞工（口語化問法）

---

## 1. 系統概覽

LaborRAG 是一套本地運行的 RAG 系統，讓一般勞工以口語方式查詢勞動法規，系統回傳精確條文引用與白話說明。核心原則：**LLM 只能引用已檢索條文，不得自行推斷法律效果。**

---

## 2. 架構總覽 (Architecture)

### 資料流

```
[laws.moj.gov.tw XML/JSON API]
        ↓ ETL Pipeline (Python)
[法條 Knowledge Atoms]  ← JSON Schema 驗證
        ↓ BGE-M3 Embedding (本地)
[PostgreSQL + pgvector]
        ↓
┌─────────────────────────────────────┐
│           Retrieval Layer           │
│  Dense:  pgvector cosine sim        │
│       +                             │
│  Sparse: PostgreSQL tsvector (BM25) │
│       ↓  RRF 融合 Top-20            │
│  BGE-Reranker-v2-m3 (cross-encoder) │
│       ↓  Top-5                      │
└────────────┬────────────────────────┘
             ↓ 條文原文 + metadata
     [Ollama LLM (Llama3.1-8B)]
             ↓
     [結構化回答 + 條文來源標註]
```

### ETL 邊界
非結構化 XML → 結構化 JSON atom → pgvector 向量化

### Reranking 介入點
粗檢索（Top-20）完成後，LLM 呼叫前。

### Knowledge Atom Schema

```json
{
  "id": "uuid",
  "law_name": "勞動基準法",
  "article_no": "24",
  "chapter": "第三章 工資",
  "content": "雇主延長勞工工作時間者，其延長工作時間之工資，依下列標準加給...",
  "keywords": ["加班費", "延長工時", "工資"],
  "last_updated": "2024-01-01",
  "parent_id": null,
  "embedding": "float[1024]  // BGE-M3 輸出維度，存於 pgvector vector(1024) 欄位"
}
```

---

## 3. 規格說明書 (Specs)

### 3.1 資料規格

| 項目 | 規格 |
|------|------|
| Chunking 策略 | 條文級切分（Article-level），每條 = 1 chunk |
| 有「項」的條文 | 整條為 parent，各項為 child chunk |
| 最大 chunk 長度 | 512 tokens（BGE-M3 上限） |
| 重疊（overlap） | 0（法條邊界明確） |
| 資料來源 | laws.moj.gov.tw 公開 XML API |
| 涵蓋法規 | 勞動基準法（含施行細則）、勞工保險條例、職業安全衛生法 |

### 3.2 性能指標

| 指標 | 目標值 |
|------|--------|
| 端對端查詢延遲 | < 2 秒（需 GPU；CPU 環境約 5–10 秒，仍可接受） |
| Top-K（最終輸出） | 5 條文 |
| Reranker 輸入 | Top-20 粗檢索結果 |
| Embedding batch size | 32 條/批 |
| 部署模式 | 本地 CLI（非 web service） |

### 3.3 評估機制（RAGAS）

| 指標 | 說明 | 目標 |
|------|------|------|
| Faithfulness | 回答完全基於檢索條文，無捏造 | ≥ 0.85 |
| Answer Relevancy | 回答與問題相關程度 | ≥ 0.80 |
| Context Precision | Top-K 中真正相關條文比例 | ≥ 0.70 |
| Context Recall | 相關條文被完整檢索到的比例 | ≥ 0.75 |

**Golden Test Set：** 20 題，覆蓋加班費、資遣費、育嬰留停、職災、勞保給付等情境，每題標注標準條號答案。

---

## 4. 架構決策紀錄 (ADR)

### ADR-001：Embedding Model — BGE-M3 vs OpenAI text-embedding-3-small

**Context:** 需 embed 中文法條，使用者問法口語化，全本地運行需求。

**Decision:** 使用 BAAI/bge-m3 本地運行。

**Status:** Accepted

**Consequences:**
- ✅ 中文理解強、支援 dense + sparse + colbert 三模式、免 API 費用、資料不出境
- ❌ 首次下載 ~2GB、CPU 環境 inference 較慢

**Rejected:** OpenAI text-embedding-3-small — 付費、資料送出境、中文語義稍弱於 BGE-M3。

---

### ADR-002：Hybrid Search（Dense + Sparse）vs 純 Vector RAG

**Context:** 勞工常用「第幾條」或精確法律術語查詢（如「資遣費」「預告期」），純語義向量無法穩定命中。

**Decision:** Dense（pgvector cosine）+ Sparse（PostgreSQL tsvector）雙路檢索，RRF（Reciprocal Rank Fusion）融合排序。

**Status:** Accepted

**Consequences:**
- ✅ 語義查詢 + 關鍵字查詢同時覆蓋；條號直查有效；無需額外工具
- ❌ 需維護兩個索引；查詢邏輯較複雜

**Rejected:** 純 Vector RAG — 精確關鍵字查詢（如「第24條」）效果不穩定。

---

### ADR-003：BGE-Reranker 介入點

**Context:** Top-20 粗檢索含語義雜訊，直接餵 LLM 增加 token 消耗且答案品質分散。

**Decision:** 粗檢索 Top-20 → BGE-Reranker-v2-m3（cross-encoder）重排 → 取 Top-5 餵 LLM。

**Status:** Accepted

**Consequences:**
- ✅ 精確度顯著提升；LLM context 精簡；rerank score 可作為可解釋性依據
- ❌ 多一次 cross-encoder inference（本地約 +200–400ms）

**Rejected:** 直接 Top-5 粗檢索 — bi-encoder 粗排召回率不足，漏掉相關條文風險高。

---

## 5. 白皮書摘要 (Whitepaper)

### 5.1 理論難點：法律語言的數位表徵

**問題一：條文語義密度高**

一條條文含多個法律概念（主體、行為、條件、效果）。例：勞基法第24條同時規範「正常工時後」「休假日」「例假日」三種加班費標準，切分子句則各自失去法律效力脈絡。

解法：以條文整體為最小 chunk 單位，禁止跨條合併。

**問題二：口語 ↔ 法律術語鴻溝**

| 口語 | 法律術語 | 條文 |
|------|---------|------|
| 炒魷魚 | 非自願離職 / 終止勞動契約 | §11–14 |
| 加班費 | 延長工時工資 | §24 |
| 試用期 | 無明文規定（法律空白） | 需說明 |
| 職災 | 職業災害 | §59 |
| 勞保 | 勞工保險 | 勞保條例 |

解法：BGE-M3 語義向量覆蓋語義距離；atom metadata 加入 `keywords` 欄位存同義詞。

### 5.2 幻覺防制策略

| 風險 | 對策 |
|------|------|
| LLM 捏造不存在條文 | 回答強制引用 atom 原文；無檢索結果回答「查無相關條文，建議諮詢律師」 |
| 條文版本過時 | atom 標記 `last_updated`，ETL 定期重新 ingest |
| 多條文矛盾解釋 | Prompt 要求列出所有相關條文原文，由使用者判斷；不由 LLM 裁決 |
| 法律空白（如試用期） | 明確告知「法無明文」，不推斷效果 |

### 5.3 冷啟動與專有名詞對齊

**冷啟動：** 法規資料完全公開，無資料冷啟動問題。系統以 laws.moj.gov.tw XML 作為唯一知識來源，20 題 golden test set 作為評估基線。

**專有名詞對齊：** 透過 ETL 階段 LLM 輔助標註 `keywords` 欄位，建立口語 → 法律術語映射表，持續人工審查擴充。

---

## 6. 技術選型總表

| 元件 | 選型 | 理由 |
|------|------|------|
| Embedding | BGE-M3 (BAAI) | 本地、中文強、三模式 |
| Reranker | BGE-Reranker-v2-m3 | 與 embedding 同家族、中文優化 |
| LLM | Llama3.1-8B (Ollama) | 本地、免費、8B 足夠摘要任務 |
| Vector DB | PostgreSQL + pgvector | 成熟、hybrid search 原生支援 |
| Retrieval | Dense + Sparse + RRF | 覆蓋語義與關鍵字查詢 |
| 評估 | RAGAS | 標準化 RAG 評估框架 |
| 資料來源 | laws.moj.gov.tw XML | 官方、公開、定期更新 |

---

## 7. 交付文件清單

- [ ] `docs/architecture/` — 系統架構圖（含 ETL 邊界、Reranking 介入點）
- [ ] `docs/specs/` — 本規格說明書完整版
- [ ] `docs/adr/` — ADR-001 至 ADR-003
- [ ] `docs/whitepaper/` — 白皮書完整版
- [ ] `schemas/` — Knowledge Atom JSON Schema
- [ ] `src/` — Python 實作（ETL、embedding、retrieval、generation）
- [ ] `tests/` — Golden test set + RAGAS 評估腳本
- [ ] `README.md` — 安裝與執行說明
