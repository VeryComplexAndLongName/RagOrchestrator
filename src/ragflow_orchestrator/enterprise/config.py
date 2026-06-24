from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class EnterpriseLLMConfig(BaseModel):
    provider: Literal["none", "openai", "ollama", "custom"] = "none"
    model: str = "gpt-4o-mini"
    max_tokens: int = 1000
    temperature: float = 0.1
    enable_tools: bool = False
    max_tool_calls: int = 3
    allowed_tools: list[str] = Field(default_factory=list)
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    timeout_seconds: int = 60


class EnterpriseIntentLLMConfig(EnterpriseLLMConfig):
    model: str = "gpt-4o-mini"


class EnterpriseAnswerLLMConfig(EnterpriseLLMConfig):
    model: str = "gpt-4o-mini"


class EnterpriseRiskConfig(BaseModel):
    enabled_by_default: bool = False
    severity_weight: float = 0.5
    likelihood_weight: float = 0.3
    impact_weight: float = 0.2


class EnterpriseToolingConfig(BaseModel):
    enable_tools: bool = True
    enable_mcp: bool = True
    strict_tool_failures: bool = False


class EnterpriseBundleConfig(BaseModel):
    default_tasks: list[str] = Field(
        default_factory=lambda: [
            "requirements_extraction",
            "compliance_check",
            "consistency_validation",
            "completeness_gap_analysis",
            "ambiguity_precision_review",
        ]
    )
    max_evidence_total: int = 24
    max_evidence_per_group: int = 8
    top_k_per_retrieval: int = 8
    group_by_metadata_key: str = "source_type"
    use_prompt_orchestrator: bool = True


class EnterprisePipelineConfig(BaseModel):
    intent_llm: EnterpriseIntentLLMConfig = Field(default_factory=EnterpriseIntentLLMConfig)
    answer_llm: EnterpriseAnswerLLMConfig = Field(default_factory=EnterpriseAnswerLLMConfig)
    risk: EnterpriseRiskConfig = Field(default_factory=EnterpriseRiskConfig)
    tooling: EnterpriseToolingConfig = Field(default_factory=EnterpriseToolingConfig)
    bundle: EnterpriseBundleConfig = Field(default_factory=EnterpriseBundleConfig)
