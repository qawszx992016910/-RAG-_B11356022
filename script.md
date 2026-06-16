# 期末專題口頭報告稿

> 自用備稿，照順序唸，每頁約 30–50 秒。

---

## 01 / 14 封面

大家好，我是 B11356022 洪偉強。
我的期末專題是「韓文學習 RAG LINE Bot」，核心技術是 LangGraph 加上 Hybrid RAG，做一個可以用自然語言問韓文的 LINE Bot。
接下來我會依序介紹動機、系統架構、知識庫、RAG 流程，最後是操作 demo 跟成果討論。

---

## 02 / 14 動機與問題定義

做這個專題的原因是：現有的韓文學習 App 大多是字典查詢，沒辦法解釋文法、出 TOPIK 練習題，也不能自然語言互動。
我想做的是一個不用下載新 App、直接在 LINE 上問問題就可以學韓文的系統。
目標使用者是台灣韓文初學者，特別是備考 TOPIK 跟靠韓劇自學的人。

---

## 03 / 14 系統架構

系統用 FastAPI 接 LINE Webhook，核心是 LangGraph 圖引擎處理問答流程。
LLM 用兩個：router 用比較便宜的 gpt-4.1-mini，回答用 gpt-4.1。
向量資料庫用 Supabase pgvector，排程推播用 APScheduler，每天早上八點跟晚上九點各推一次。

---

## 04 / 14 LangGraph 工作流程

整個問答流程有六個主要節點。
① Router 先判斷使用者想問什麼，② Retrieve 做 Hybrid RAG 檢索，③ Sufficiency 判斷撈到的資料夠不夠，④ Generate 生成回答，⑤ Judge 對回答做四軸評分，如果品質不夠，⑥ Reflect 會重新生成，最多一次。
Judge 評的四個軸是：事實有沒有依據、有沒有引用錯、格式對不對、不確定的有沒有說明。

---

## 05 / 14 Skill 設計

系統設計了八個 Skill，每個 Skill 有自己的 SKILL.md 規則檔，定義輸出格式跟禁止行為。
Router 是雙層設計：先用關鍵詞快速判斷，沒命中才呼叫 LLM。
八個 Skill 裡有三個會用到 RAG 知識庫：vocabulary_master、grammar_guide、topik_tutor；其他的像翻譯、對話練習是純 LLM 生成。

---

## 06 / 14 知識庫設計

知識庫總共 845 個 chunks，分三個 category：grammar_patterns 775 個、vocabulary 41 個、topik_questions 10 個。
建置流程是：用 Playwright 爬蟲抓 howtostudykorean.com，轉成 Markdown，切段後用 OpenAI text-embedding-3-small 向量化，存進 Supabase pgvector。
去重用 SHA256 content_hash，所以修改 MD 重新 ingest 不會產生重複 chunk。

---

## 07 / 14 RAG Pipeline

RAG 流程分兩段。
Ingestion 是爬蟲到向量化到存入，剛才講過了。
Retrieval 是：使用者問題先做 Intent routing，再用 Multi-seed 查詢，也就是把原始問題加上子問題分開查，然後用 Hybrid + RRF 兩階段排序，最後取 Top-4 chunks 給 generator。

---

## 08 / 14 資料來源

資料來源主要三個：第一是爬 howtostudykorean.com，爬了 Unit 1 的 L1-8 跟 Unit 2 的 L26-33，共 16 頁，產生了 775 個 grammar chunks。
第二是我手工整理的單字 MD 檔跟 TOPIK 練習題，每份單字檔都有定義、발음、例句和易混淆詞說明。
第三是課程提供的基礎架構，我在這基礎上重寫了 skill 設計跟整個知識庫。

---

## 09 / 14 系統操作說明

啟動只要三步：pip install、uvicorn 跑 FastAPI、ngrok 開通道，然後把 Webhook URL 填進 LINE Developers。
.env 需要填 OpenAI key、LINE token 跟 Supabase 連線資訊。
可以問的範圍包括：查單字、問文法、要 TOPIK 練習題、中韓翻譯、對話練習，還有每天自動推播。

---

## 10 / 14 操作截圖 1：vocabulary_master

這頁示範查單字的情境。
使用者問「모레는 무슨 뜻」，keyword「뜻」命中，路由到 vocabulary_master，RAG 撈到 모레.md 的易混淆詞 section。
系統會給出定義、발음、例句，還有跟 모래（沙子）的發音差異說明，這是 SKILL.md 規定的：只有발음或拼寫相近的才算易混淆，同義詞不算。

---

## 11 / 14 操作截圖 2：grammar + topik

左邊是 grammar_guide 的示範，問「아/어서 怎麼用」，會給出文法結構說明跟三個以上例句，而且嚴格只用 grammar_patterns category 的 chunk，不會混入詞彙說明。
右邊是 topik_tutor，出一題句型填空，四個選項都是同語法類型，最後會解析答案。

---

## 12 / 14 後台驗證：Hybrid RAG

這頁解釋 Hybrid RRF 的實作。
第一階段是每個 seed 查詢計算 Hybrid 分數：0.7 乘以 cosine 向量分數加上 0.3 乘以 BM25 關鍵詞分數，這樣可以同時命中語意相似和關鍵詞完全吻合的 chunk。
第二階段是 Multi-seed RRF 融合：把原始問題跟子問題的結果用 Reciprocal Rank Fusion 合併，公式是各 seed 排名的倒數加總，k 等於 60，這樣可以鈍化極端分數、讓多個 seed 都同意的 chunk 排得更前面。

---

## 13 / 14 成果討論

成功的地方有四個：易混淆詞辨識正確、文法說明格式精準、TOPIK 出題類型正確、全部八個 skill 測試通過。
限制是：文法只涵蓋到 Unit 2，TOPIK 練習題是手工整理非真題，而且沒有多輪對話記憶。
過程中解了六個以上的 bug，比較值得講的是：judge 原本不分 skill 類型全部都審查，導致翻譯、對話這種不用 RAG 的 skill 也跳品質警告，後來加了 SKIP_JUDGE_SKILLS 白名單解決。

---

## 14 / 14 結論

總結：建置了 845 個知識庫 chunks，設計並測試了八個 skill，爬了 16 頁文法資料，修了六個以上重大 bug。
這個專題讓我實際走過完整的 RAG 流程，從資料整理到向量化到檢索到生成，還有 LangGraph reflection 回路的設計。
未來可以繼續擴充知識庫、加多輪對話記憶，以及改用 Railway 做穩定部署。以上，謝謝。
