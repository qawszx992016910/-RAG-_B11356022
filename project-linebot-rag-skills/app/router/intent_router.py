from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Protocol

from app.router.emotion_detector import detect_emotion
from app.router.prompts import render_router_prompt
from app.router.schemas import EmotionState, ResponseMode, RouterResult, SkillId


VOCAB_KEYWORDS = ("單字", "詞", "什麼意思", "怎麼用", "差別", "韓文字", "뜻", "의미")
GRAMMAR_KEYWORDS = ("文法", "句型", "語尾", "助詞", "grammar", "고 싶다", "아/어서", "으로")
TOPIK_KEYWORDS = ("topik", "考試", "試題", "備考", "토픽")
READING_KEYWORDS = ("解析", "分析", "文法結構", "這句", "這段", "段落解析", "句型分析", "逐詞")
CONVERSATION_KEYWORDS = ("練習", "會話", "對話", "情境", "口說")
TRANSLATE_TO_KO_KEYWORDS = ("翻成韓文", "韓文怎麼說", "中翻韓", "用韓文說", "韓語怎麼說")
TRANSLATE_TO_ZH_KEYWORDS = ("翻成中文", "韓翻中", "這句翻")
# Korean syllable block detection: any char in Hangul range
import re as _re
_HANGUL_RE = _re.compile(r"[가-힣ㄱ-ㆎ]")


class RouterLLM(Protocol):
    async def complete(self, prompt: str) -> str:
        ...


@dataclass
class IntentRouter:
    llm: RouterLLM | None = None
    confidence_threshold: float = 0.55

    async def route_message(self, user_input: str, recent_history: str) -> RouterResult:
        emotion = detect_emotion(user_input)
        if self.llm is None:
            return self._heuristic_route(user_input, emotion)

        try:
            prompt = render_router_prompt(user_input, recent_history)
            raw_output = await self.llm.complete(prompt)
            parsed = self._parse_router_output(raw_output)
            result = RouterResult.model_validate(parsed)
            return self._normalize_result(result, user_input, emotion)
        except Exception:
            return self._heuristic_route(user_input, emotion)

    def _normalize_result(
        self,
        result: RouterResult,
        user_input: str,
        fallback_emotion: EmotionState,
    ) -> RouterResult:
        normalized = result.model_copy(
            update={
                "rag_query": result.rag_query.strip() or user_input.strip(),
                "emotion_state": result.emotion_state or fallback_emotion,
            }
        )
        if normalized.confidence < self.confidence_threshold:
            return self._heuristic_route(user_input, fallback_emotion)
        return normalized

    def _parse_router_output(self, raw_output: str) -> dict[str, object]:
        stripped = raw_output.strip()
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start == -1 or end == -1:
                raise
            return json.loads(stripped[start : end + 1])

    def _heuristic_route(self, user_input: str, emotion: EmotionState) -> RouterResult:
        lowered = user_input.lower()
        has_hangul = bool(_HANGUL_RE.search(user_input))
        has_chinese = any("一" <= c <= "鿿" for c in user_input)

        if any(kw in lowered for kw in TOPIK_KEYWORDS):
            return RouterResult.fallback(
                user_input,
                target_skill="topik_tutor",
                emotion_state=emotion,
                response_mode="structured",
                is_rag_required=True,
                rag_categories=["topik_questions"],
                confidence=0.75,
            )
        # Translation: Chinese → Korean
        if any(kw in lowered for kw in TRANSLATE_TO_KO_KEYWORDS) or (has_chinese and "韓" in user_input and not has_hangul):
            return RouterResult.fallback(
                user_input,
                target_skill="translator",
                emotion_state=emotion,
                response_mode="structured",
                is_rag_required=False,
                confidence=0.8,
            )
        # Translation: Korean → Chinese (explicit request)
        if any(kw in lowered for kw in TRANSLATE_TO_ZH_KEYWORDS) and has_hangul:
            return RouterResult.fallback(
                user_input,
                target_skill="translator",
                emotion_state=emotion,
                response_mode="structured",
                is_rag_required=False,
                confidence=0.8,
            )
        # Korean sentence grammar analysis (longer Korean input = analysis, not vocab lookup)
        if has_hangul and any(kw in lowered for kw in READING_KEYWORDS):
            return RouterResult.fallback(
                user_input,
                target_skill="reading_helper",
                emotion_state=emotion,
                response_mode="structured",
                is_rag_required=False,
                confidence=0.75,
            )
        # Long Korean input (sentence) without translation request → analysis
        if has_hangul and len(user_input.strip()) > 10 and not any(kw in lowered for kw in VOCAB_KEYWORDS):
            return RouterResult.fallback(
                user_input,
                target_skill="reading_helper",
                emotion_state=emotion,
                response_mode="structured",
                is_rag_required=False,
                confidence=0.65,
            )
        if any(kw in lowered for kw in GRAMMAR_KEYWORDS):
            return RouterResult.fallback(
                user_input,
                target_skill="grammar_guide",
                emotion_state=emotion,
                response_mode="structured",
                is_rag_required=True,
                rag_categories=["grammar_patterns"],
                confidence=0.75,
            )
        if any(kw in lowered for kw in CONVERSATION_KEYWORDS):
            return RouterResult.fallback(
                user_input,
                target_skill="conversation_coach",
                emotion_state=emotion,
                response_mode="brief",
                is_rag_required=False,
                confidence=0.7,
            )
        # Bare Korean word/short input → vocabulary lookup
        if has_hangul and (any(kw in lowered for kw in VOCAB_KEYWORDS) or len(user_input.strip()) <= 10):
            return RouterResult.fallback(
                user_input,
                target_skill="vocabulary_master",
                emotion_state=emotion,
                response_mode="structured",
                is_rag_required=True,
                rag_categories=["vocabulary", "confusable_pairs"],
                confidence=0.7,
            )
        return RouterResult.fallback(
            user_input,
            target_skill="general_chat",
            emotion_state=emotion,
            response_mode="brief",
            confidence=0.5,
        )
