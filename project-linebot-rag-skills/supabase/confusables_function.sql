-- find_similar_words: 用 pg_trgm 找字形相似的詞條
-- 用於 detect_confusables node，在回覆單字問題時自動提示易混淆字。
-- 需要先在 Supabase 啟用 pg_trgm extension:
--   CREATE EXTENSION IF NOT EXISTS pg_trgm;
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
