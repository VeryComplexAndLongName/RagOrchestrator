from __future__ import annotations

from datetime import date

from ragflow_orchestrator.enterprise import (
    ACLAwareRetrieverAdapter,
    EnterpriseAnswerLLMConfig,
    EnterpriseIntentLLMConfig,
    EnterprisePipeline,
    EnterprisePipelineConfig,
    ReviewBundleRequest,
    ReviewTask,
)
from ragflow_orchestrator.models import BaseChunk, RetrievalResult
from ragflow_orchestrator.orchestrator import RAGOrchestrator
from ragflow_orchestrator.presets import document_preset


class DummyRetriever:
    def retrieve(self, question: str, top_k: int = 5, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
        _ = question, top_k, filters
        chunk = BaseChunk(
            id="c1",
            text="Document clause says equipment must have grounding and emergency stop.",
            metadata={
                "source_type": "normative",
                "document_id": "doc-1",
                "version_id": "v1",
                "page": "12",
                "clause_path": "5.2.3",
                "valid_to": "2099-12-31",
            },
            source_id="doc-1",
            chunk_index=0,
        )
        return [RetrievalResult(chunk=chunk, score=0.91, provider="dummy")]


class DummyProvider:
    name = "dummy"

    def ensure_schema(self, vector_dim: int) -> None:
        _ = vector_dim

    def upsert_chunks(self, chunks: list[BaseChunk]) -> None:
        _ = chunks

    def delete_chunks(self, chunk_ids: list[str], soft_delete: bool = True) -> None:
        _ = chunk_ids, soft_delete

    def retrieve(self, query):
        _ = query
        chunk = BaseChunk(id="d1", text="dummy", metadata={}, source_id="d", chunk_index=0)
        return [RetrievalResult(chunk=chunk, score=0.5, provider=self.name)]

    def healthcheck(self) -> bool:
        return True


class DummyEmbedder:
    dimensions = 4

    def embed(self, text: str) -> list[float]:
        _ = text
        return [0.1, 0.2, 0.3, 0.4]

    def embed_many(self, texts):
        return [self.embed(text) for text in texts]


class IntentLLMStub:
    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        _ = prompt, model, max_tokens, temperature
        return '{"tasks": ["risk_assessment", "project_error_analysis"], "tool_calls": [], "notes": "ok"}'


class AnswerLLMStub:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def generate(self, prompt: str, model: str, max_tokens: int, temperature: float) -> str:
        _ = model, max_tokens, temperature
        self.prompts.append(prompt)
        if "planning tool calls for answer generation" in prompt.lower():
            return (
                '{"tool_calls": ['
                '{"name": "score_risks", "args": {"severity": "high", "likelihood": 0.9, "impact": 0.8}}'
                "]}"
            )
        return "Generated answer"


def test_review_bundle_runs_with_defaults() -> None:
    pipeline = EnterprisePipeline(query_retriever=DummyRetriever())
    result = pipeline.review_bundle(
        ReviewBundleRequest(
            session_id="s1",
            query_text="Check compliance and find possible issues",
            department_principals=["dept_a"],
            as_of_date=date.today(),
            document_ids=["doc-1"],
        )
    )

    assert result.executed_tasks
    assert result.grouped_evidence
    assert result.prompt
    assert result.llm_answer


def test_review_bundle_respects_optional_risk_flag() -> None:
    pipeline = EnterprisePipeline(query_retriever=DummyRetriever())
    result = pipeline.review_bundle(
        ReviewBundleRequest(
            session_id="s2",
            query_text="Assess requirements",
            requested_tasks=[ReviewTask.REQUIREMENTS_EXTRACTION],
            enable_risk_assessment=False,
        )
    )

    assert result.risk_assessment.enabled is False
    assert result.risk_assessment.entries == []


def test_acl_aware_adapter_forwards_acl_principals() -> None:
    captured = {}

    class CapturingProvider(DummyProvider):
        def retrieve(self, query):
            captured["acl_principals"] = query.acl_principals
            return super().retrieve(query)

    adapter = ACLAwareRetrieverAdapter(provider=CapturingProvider(), embedder=DummyEmbedder())
    adapter.retrieve("hello", top_k=3, filters={"acl_principals": ["dept_a", "dept_b"], "source_type": "normative"})

    assert captured["acl_principals"] == ["dept_a", "dept_b"]


def test_from_orchestrator_constructor_and_tool_call() -> None:
    preset = document_preset()
    orchestrator = RAGOrchestrator(
        provider=DummyProvider(),
        embedder=DummyEmbedder(),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )
    pipeline = EnterprisePipeline.from_orchestrator(
        orchestrator,
        config=EnterprisePipelineConfig(),
    )

    response = pipeline.call_tool(
        name="score_risks",
        request_id="req-1",
        args={"severity": "high", "likelihood": 0.9, "impact": 0.8},
    )

    assert response["ok"] is True
    assert response["priority_score"] > 0.0


def test_llm_intent_analysis_selects_tasks() -> None:
    config = EnterprisePipelineConfig(
        intent_llm=EnterpriseIntentLLMConfig(provider="custom"),
        answer_llm=EnterpriseAnswerLLMConfig(provider="none"),
    )
    pipeline = EnterprisePipeline(
        query_retriever=DummyRetriever(),
        config=config,
        intent_llm_client=IntentLLMStub(),
    )

    result = pipeline.review_bundle(
        ReviewBundleRequest(
            session_id="intent-1",
            query_text="Проведи анализ",
            requested_tasks=[],
        )
    )

    assert result.executed_tasks == [ReviewTask.RISK_ASSESSMENT, ReviewTask.PROJECT_ERROR_ANALYSIS]


def test_answer_llm_can_plan_and_use_tools() -> None:
    answer_client = AnswerLLMStub()
    config = EnterprisePipelineConfig(
        intent_llm=EnterpriseIntentLLMConfig(provider="none"),
        answer_llm=EnterpriseAnswerLLMConfig(
            provider="custom",
            enable_tools=True,
            allowed_tools=["score_risks"],
            max_tool_calls=1,
        ),
    )
    pipeline = EnterprisePipeline(
        query_retriever=DummyRetriever(),
        config=config,
        answer_llm_client=answer_client,
    )

    result = pipeline.review_bundle(
        ReviewBundleRequest(
            session_id="ans-1",
            query_text="Сделай вывод и приоритеты",
            requested_tasks=[ReviewTask.COMPLIANCE_CHECK],
        )
    )

    assert result.llm_answer == "Generated answer"
    assert len(answer_client.prompts) >= 2
    assert "Tool execution results" in answer_client.prompts[-1]
