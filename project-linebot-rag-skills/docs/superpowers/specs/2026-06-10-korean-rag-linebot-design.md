# 韓文學習 RAG LINE Bot — 設計文件

**日期：** 2026-06-10
**專案：** project-linebot-rag-skills
**狀態：** 設計確認中

---

## 1. 目標

打造一個 LINE Bot，以 RAG 驅動韓文學習問答，並提供每日自動推播。
核心差異化：不是通用韓文 AI，而是**從特定教材爬取的有限知識庫**，確保回答可追溯來源。

---

## 2. 架構總覽

```
Playwright Crawler
  └─ TOPIK 歷屆試題 / TTMIK 課程 / 세종학당 材料 / 국립국어원 字典 / PDF 課本
       ↓ ingest_markdown.py
       ↓ text-embedding-3-small
pgvector（private_knowledge）+ pg_trgm

LINE 用戶訊息
  ↓
FastAPI Webhook
  ↓
LangGraph StateGraph（reflection variant）
  ├─ route          → Intent Router（gpt-4.1-mini）→ skill 分類
  ├─ extract        → Feature Extractor → 目標詞、文法型、難度
  ├─ confusable     → pg_trgm 相似詞偵測（與 retrieve 並行）
  ├─ expand_seeds   → Multi-seed 展開（型態 + 用法 + 例句）
  ├─ retrieve × N  → Hybrid RAG（vector + keyword，metadata filter by category）
  ├─ fuse           → RRF Score Fusion
  ├─ sufficiency    → 資料不足 → clarify；充足 → 繼續
  ├─ build_contract → Answer Contract JSON（word, definition, examples, level, confusables）
  ├─ render         → Narrative 生成（gpt-4.1，受限 prompt）
  ├─ judge          → 4 軸評分（groundedness / citation / format / level_match）
  │     ├─ fail → render（retry ≤ 2）
  │     └─ pass → push
  └─ push           → LINE Push API

APScheduler（FastAPI startup）
  ├─ 08:00 每日單字推播
  ├─ 08:05 每日文法推播
  └─ 21:00 今日 TOPIK 題推播
```

---

## 3. 知識庫規劃

### 3.1 Categories

| Category | 來源 | 爬法 | 數量目標 |
|----------|------|------|----------|
| `vocabulary` | 국립국어원 한국어기초사전（KED） | Playwright | 3,000+ 詞條 |
| `grammar_patterns` | TTMIK 免費課程頁 | Playwright | 200+ 文法型 |
| `topik_questions` | TOPIK 官方歷屆試題（공개 문제） | Playwright | 500+ 題 |
| `reading_passages` | 세종학당 教材 / Easy Korean News | Playwright | 100+ 篇 |
| `textbook_passages` | 用戶自有 PDF 課本 | pdfplumber | 視 PDF 數量 |
| `confusable_pairs` | 手工策展 + pg_trgm 自動偵測 | 手工 markdown | 50+ 對 |
| `cultural_notes` | 세종학당 文化材料 | Playwright | 50+ 筆 |

### 3.2 Frontmatter 規範

```yaml
---
category: grammar_patterns
source_url: https://talktomeinkorean.com/lessons/...
content_hash: sha256-auto
difficulty: TOPIK_2   # beginner | TOPIK_1 | TOPIK_2 | advanced
tags: [아/어서, 因為, 連接語尾]
language: ko
---
```

`difficulty` 欄位用於 metadata filter，讓 Router 依用戶等級篩選。

---

## 4. 六個 Skill 定義

| skill_id | 名稱 | 觸發情境 | RAG Category | 備註 |
|----------|------|----------|--------------|------|
| `vocabulary_master` | 單字大師 | 「XX 是什麼意思」「XX 怎麼用」 | `vocabulary`, `confusable_pairs` | 自動觸發相似詞偵測 |
| `grammar_guide` | 文法指南 | 「XX 文法怎麼用」「XX vs XX」 | `grammar_patterns` | Multi-seed：型態+用法+例句 |
| `topik_tutor` | TOPIK 教練 | 「出題考我」「TOPIK 幾級考這個」 | `topik_questions` | 含答案解析 |
| `reading_helper` | 閱讀助手 | 「幫我看這段」「這篇什麼意思」 | `reading_passages`, `textbook_passages` | 支援用戶貼文字 |
| `conversation_coach` | 會話教練 | 「幫我練對話」「怎麼說 XX」 | LLM 直接（無 RAG） | 角色扮演情境 |
| `daily_scheduler` | 每日推播 | 排程觸發 / 「今天推什麼」 | 跨 category | 含防重複邏輯 |

---

## 5. 相似詞自動偵測

### 5.1 流程

```
用戶：「모래가 뭐예요？」
         ↓
Feature Extractor → extracted_word = "모래"
         ↓ 並行 fan-out
┌─────────────────┐    ┌──────────────────────────────────┐
│ 主 Retrieval    │    │ Confusable Detection node        │
│ vector search   │    │ SELECT word, meaning             │
│ "모래"          │    │ FROM vocabulary                  │
│                 │    │ WHERE similarity(word,'모래')>0.6 │
│                 │    │ AND word != '모래'               │
│                 │    │ → 找到：모레                     │
└─────────────────┘    └──────────────────────────────────┘
         ↓ fan-in
Answer Contract：
{
  "word": "모래",
  "definition": "...",
  "examples": [...],
  "confusables": [{"word":"모레","meaning":"後天","note":"ㅐ vs ㅔ"}]
}
         ↓
Bot 回覆自動含易混淆提示
```

### 5.2 相似度閾值

| 閾值 | 效果 |
|------|------|
| > 0.8 | 幾乎同字（排除） |
| 0.5–0.8 | 易混淆對（顯示警告）|
| < 0.5 | 不相關（忽略） |

---

## 6. 每日推播系統

### 6.1 推播排程

```python
# APScheduler 掛在 FastAPI lifespan
scheduler.add_job(push_daily_vocab,    'cron', hour=8,  minute=0)
scheduler.add_job(push_daily_grammar,  'cron', hour=8,  minute=5)
scheduler.add_job(push_topik_question, 'cron', hour=21, minute=0)
```

### 6.2 防重複機制

新增 Supabase 表：

```sql
CREATE TABLE push_history (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  line_user_id text NOT NULL,
  content_id uuid NOT NULL REFERENCES private_knowledge(id),
  push_type text NOT NULL,  -- vocab | grammar | topik | reading
  pushed_at timestamptz DEFAULT now()
);
CREATE INDEX ON push_history(line_user_id, push_type, pushed_at);
```

檢索時排除近 90 天推過的內容：

```sql
WHERE id NOT IN (
  SELECT content_id FROM push_history
  WHERE line_user_id = :uid
  AND pushed_at > NOW() - INTERVAL '90 days'
)
```

全部推完 → 自動重置（刪除 90 天以前紀錄，重新循環）。

### 6.3 推播格式範例

```
📖 今日單字

모래 [mo-rae]
명사 | TOPIK 初級

✏️ 意思：沙子
📝 例句：
  • 모래사장에서 놀았어요（在沙灘玩了）
  • 모래바람이 불어요（沙塵風在吹）

⚠️ 易混淆：모레（後天）
  ㅐ vs ㅔ，發音幾乎一樣，意思完全不同！
```

---

## 7. LangGraph 節點調整

### 新增節點

| 節點 | 說明 |
|------|------|
| `detect_confusables` | pg_trgm 相似詞查詢，與主 retrieve 並行 |
| `daily_push_selector` | 從知識庫選今日推播內容（含 push_history 過濾）|

### Judge 4 軸調整（韓文學習版）

| 軸 | 說明 | 門檻 |
|----|------|------|
| `groundedness` | 回答是否有 RAG 來源支撐 | 0.7 |
| `citation_fidelity` | 例句是否真實出現在知識庫 | 0.7 |
| `level_match` | 難度是否符合問題等級 | 0.7 |
| `confusable_coverage` | 有易混淆詞時是否有提醒 | 0.6 |

---

## 8. 爬蟲規劃

### 目標網站與策略

| 網站 | URL | 策略 |
|------|-----|------|
| 한국어기초사전（KED） | krdict.korean.go.kr | 逐詞條爬，帶例句 |
| TOPIK 官方 | topik.go.kr | 歷屆試題 PDF 下載 → pdfplumber |
| TTMIK | talktomeinkorean.com/lessons | 課程列表 → 逐課爬 |
| 세종학당 | sejonghakdang.org | 교재 免費材料頁 |
| Easy Korean News | easykorean.news | 分級閱讀文章 |

### 爬蟲節流設定

```python
# site_rules.py
SITE_RULES = {
    "krdict.korean.go.kr": {"delay": 2.0, "respect_robots": True},
    "talktomeinkorean.com": {"delay": 3.0, "respect_robots": True},
}
```

---

## 9. 技術棧對應確認

| 技術 | 用途 | 狀態 |
|------|------|------|
| FastAPI | LINE Webhook | ✅ 已有 |
| LangGraph reflection | 完整 RAG 流程 | ✅ 已有 |
| pgvector | 語意檢索 | ✅ 已有 |
| pg_trgm | 相似詞偵測 + 關鍵字搜 | ✅ 已有 |
| OpenAI text-embedding-3-small | 向量化 | ✅ 已有 |
| gpt-4.1-mini | Router | ✅ 已有 |
| gpt-4.1 | Generator | ✅ 已有 |
| Hybrid RAG | vector + keyword fusion | ✅ 已有 |
| Metadata filter | category / difficulty 篩選 | ✅ 已有 |
| Multi-seed + RRF | 展開多 seed 並行檢索 | ✅ 已有 |
| Sufficiency check | 資料不足誠實追問 | ✅ 已有 |
| Two-stage generator | Contract → Narrative | ✅ 已有 |
| LLM-as-Judge | 4 軸品質把關 | ✅ 已有 |
| Playwright | 爬蟲 | ✅ 已有（待設定規則）|
| pdfplumber | PDF 課本 ingest | ✅ 已有 |
| APScheduler | 定時推播 | ⚠️ 需新增 |
| push_history 表 | 防重複推播 | ⚠️ 需新增 |
| detect_confusables node | 相似詞自動偵測 | ⚠️ 需新增 |

---

## 10. 不做（明確排除）

- VOCAB.STUDIO 串聯（後續更新）
- 圖片 OCR 識別韓文
- 語音訊息處理
- 多用戶管理系統
- Web UI / API channel

---

## 11. 成功標準

- [ ] 5 個網站完成爬蟲並入庫
- [ ] vocabulary_master：問任一常見韓文詞能回答 + 自動顯示易混淆詞
- [ ] grammar_guide：問文法型能回答含例句，來源可追溯
- [ ] topik_tutor：能出歷屆試題並解析答案
- [ ] 每日推播三條（單字、文法、TOPIK題）正常運作
- [ ] push_history 防重複：同一詞 90 天內不重複推
- [ ] Judge pass rate ≥ 80%
