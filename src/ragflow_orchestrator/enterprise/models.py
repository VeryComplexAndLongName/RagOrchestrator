from __future__ import annotations

from datetime import date
from enum import Enum

from pydantic import BaseModel, Field


class ReviewTask(str, Enum):
    REQUIREMENTS_EXTRACTION = "requirements_extraction"
    COMPLIANCE_CHECK = "compliance_check"
    CONSISTENCY_VALIDATION = "consistency_validation"
    COMPLETENESS_GAP_ANALYSIS = "completeness_gap_analysis"
    AMBIGUITY_PRECISION_REVIEW = "ambiguity_precision_review"
    PROJECT_ERROR_ANALYSIS = "project_error_analysis"
    CHANGE_IMPACT_DIFF = "change_impact_diff"
    RISK_ASSESSMENT = "risk_assessment"


class ReviewBundleRequest(BaseModel):
    session_id: str
    query_text: str
    document_ids: list[str] = Field(default_factory=list)
    department_principals: list[str] = Field(default_factory=list)
    as_of_date: date | None = None
    requested_tasks: list[ReviewTask] = Field(default_factory=list)
    enable_risk_assessment: bool | None = None
    extra_context: dict[str, str] = Field(default_factory=dict)


class EvidenceItem(BaseModel):
    id: str
    group: str
    content: str
    score: float | None = None
    attribution: str = ""
    metadata: dict[str, str] = Field(default_factory=dict)


class TaskOutput(BaseModel):
    task: ReviewTask
    summary: str
    findings: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class RiskEntry(BaseModel):
    title: str
    severity: str
    likelihood: float
    impact: float
    confidence: float
    priority_score: float
    mitigation: str
    evidence_ids: list[str] = Field(default_factory=list)


class RiskAssessmentResult(BaseModel):
    enabled: bool
    entries: list[RiskEntry] = Field(default_factory=list)


class ReviewBundleResult(BaseModel):
    requested_tasks: list[ReviewTask]
    executed_tasks: list[ReviewTask]
    grouped_evidence: dict[str, list[EvidenceItem]]
    task_outputs: list[TaskOutput]
    risk_assessment: RiskAssessmentResult
    prompt: str
    llm_answer: str
