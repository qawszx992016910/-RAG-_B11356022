from pathlib import Path
from app.skills.registry import SkillRegistry

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
