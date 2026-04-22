from __future__ import annotations

from pydantic import BaseModel, Field


class ImportanceScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reason: str = Field(default="")


class SummaryUpdate(BaseModel):
    summary: str = Field(description="Updated compact global summary.")


class SilentTurnDecision(BaseModel):
    compressed_summary: str = Field(default="")
    write_daily: bool = Field(default=False)
    daily_note: str = Field(default="")
    write_evergreen: bool = Field(default=False)
    evergreen_note: str = Field(default="")


class DailyLogRecordDecision(BaseModel):
    should_write: bool = Field(default=False)
    summary: str = Field(default="")
    category: str = Field(default="conversation")
    confidence: float = Field(default=0.7, ge=0, le=1)
    facts: list[str] = Field(default_factory=list)
    decisions: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    user_preferences: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class ReflectionDecision(BaseModel):
    should_write: bool = Field(default=False)
    lesson: str = Field(default="")
    trigger: str = Field(default="")
    recommendation: str = Field(default="")
    confidence: float = Field(default=0.7, ge=0, le=1)
    tags: list[str] = Field(default_factory=list)


class EvergreenDiaryDecision(BaseModel):
    should_write: bool = Field(default=False)
    note: str = Field(default="")
    confidence: float = Field(default=0.7, ge=0, le=1)
    tags: list[str] = Field(default_factory=list)


DAILY_LOG_SCHEMA_VERSION = 2
EVERGREEN_SCHEMA_VERSION = 2
EVERGREEN_USER_MODULE = "user"
EVERGREEN_REFLECTION_MODULE = "agent_reflections"
