from __future__ import annotations

import time
from pathlib import Path

from ragflow_orchestrator.enterprise import (
    EnterpriseAnswerLLMConfig,
    EnterpriseIntentLLMConfig,
    EnterprisePipeline,
    EnterprisePipelineConfig,
    ReviewBundleRequest,
)
from ragflow_orchestrator.enterprise.tools import ToolContext
from ragflow_orchestrator.models import BaseChunk, RetrievalResult


class TracingEnterprisePipeline(EnterprisePipeline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.last_prompt_source: str = "unknown"
        self._reset_timing()

    def _reset_timing(self) -> None:
        self.timing: dict[str, object] = {
            "llm_calls_total": 0,
            "llm_time_total_s": 0.0,
            "prompt_tokens_total": 0,
            "completion_tokens_total": 0,
            "tokens_total": 0,
            "intent_calls": 0,
            "intent_time_s": 0.0,
            "answer_calls": 0,
            "answer_time_s": 0.0,
            "planner_calls": 0,
            "planner_time_s": 0.0,
            "llm_calls": [],
            "review_bundle_time_s": 0.0,
        }

    def review_bundle(self, request_payload: ReviewBundleRequest):  # type: ignore[override]
        self._reset_timing()
        started = time.perf_counter()
        result = super().review_bundle(request_payload)
        self.timing["review_bundle_time_s"] = time.perf_counter() - started
        return result

    def _build_prompt(self, req, tasks, grouped, task_outputs, risk_result) -> str:  # type: ignore[no-untyped-def]
        if self.config.bundle.use_prompt_orchestrator and self._check_prompt_orchestrator_available():
            built = self._build_prompt_via_prompt_orchestrator(req=req, grouped=grouped)
            if built:
                self.last_prompt_source = "prompt_orchestrator"
                return built

        self.last_prompt_source = "fallback"
        was_enabled = self.config.bundle.use_prompt_orchestrator
        self.config.bundle.use_prompt_orchestrator = False
        try:
            return super()._build_prompt(req, tasks, grouped, task_outputs, risk_result)
        finally:
            self.config.bundle.use_prompt_orchestrator = was_enabled

    def _call_llm_text(self, *, prompt: str, cfg, custom_client):  # type: ignore[override,no-untyped-def]
        started = time.perf_counter()
        result = super()._call_llm_text(prompt=prompt, cfg=cfg, custom_client=custom_client)
        duration = time.perf_counter() - started

        provider = str(getattr(cfg, "provider", "unknown"))
        model = str(getattr(cfg, "model", "unknown"))
        meta = getattr(self, "_last_llm_meta", {})
        prompt_tokens = int(meta.get("prompt_tokens", 0)) if isinstance(meta, dict) else 0
        completion_tokens = int(meta.get("completion_tokens", 0)) if isinstance(meta, dict) else 0
        total_tokens = int(meta.get("total_tokens", 0)) if isinstance(meta, dict) else 0
        endpoint = str(meta.get("endpoint", "unknown")) if isinstance(meta, dict) else "unknown"
        prompt_preview = prompt[:120].replace("\n", " ")
        phase = "answer"
        lowered = prompt.lower()
        if "analyze enterprise review intent" in lowered:
            phase = "intent"
            self.timing["intent_calls"] = int(self.timing["intent_calls"]) + 1
            self.timing["intent_time_s"] = float(self.timing["intent_time_s"]) + duration
        elif "planning tool calls for answer generation" in lowered:
            phase = "planner"
            self.timing["planner_calls"] = int(self.timing["planner_calls"]) + 1
            self.timing["planner_time_s"] = float(self.timing["planner_time_s"]) + duration
        else:
            self.timing["answer_calls"] = int(self.timing["answer_calls"]) + 1
            self.timing["answer_time_s"] = float(self.timing["answer_time_s"]) + duration

        self.timing["llm_calls_total"] = int(self.timing["llm_calls_total"]) + 1
        self.timing["llm_time_total_s"] = float(self.timing["llm_time_total_s"]) + duration
        self.timing["prompt_tokens_total"] = int(self.timing["prompt_tokens_total"]) + prompt_tokens
        self.timing["completion_tokens_total"] = int(self.timing["completion_tokens_total"]) + completion_tokens
        self.timing["tokens_total"] = int(self.timing["tokens_total"]) + total_tokens
        llm_calls = self.timing["llm_calls"]
        assert isinstance(llm_calls, list)
        llm_calls.append(
            {
                "phase": phase,
                "provider": provider,
                "model": model,
                "endpoint": endpoint,
                "duration_s": round(duration, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_preview": prompt_preview,
            }
        )
        return result

    def _generate_answer(self, prompt: str, task_outputs, req):  # type: ignore[override,no-untyped-def]
        started = time.perf_counter()
        result = super()._generate_answer(prompt, task_outputs, req)
        duration = time.perf_counter() - started

        cfg = self.config.answer_llm
        if str(getattr(cfg, "provider", "none")) == "none":
            return result

        provider = str(getattr(cfg, "provider", "unknown"))
        model = str(getattr(cfg, "model", "unknown"))
        meta = getattr(self, "_last_llm_meta", {})
        prompt_tokens = int(meta.get("prompt_tokens", 0)) if isinstance(meta, dict) else 0
        completion_tokens = int(meta.get("completion_tokens", 0)) if isinstance(meta, dict) else 0
        total_tokens = int(meta.get("total_tokens", 0)) if isinstance(meta, dict) else 0
        endpoint = str(meta.get("endpoint", "unknown")) if isinstance(meta, dict) else "unknown"

        self.timing["answer_calls"] = int(self.timing["answer_calls"]) + 1
        self.timing["answer_time_s"] = float(self.timing["answer_time_s"]) + duration
        self.timing["llm_calls_total"] = int(self.timing["llm_calls_total"]) + 1
        self.timing["llm_time_total_s"] = float(self.timing["llm_time_total_s"]) + duration
        self.timing["prompt_tokens_total"] = int(self.timing["prompt_tokens_total"]) + prompt_tokens
        self.timing["completion_tokens_total"] = int(self.timing["completion_tokens_total"]) + completion_tokens
        self.timing["tokens_total"] = int(self.timing["tokens_total"]) + total_tokens

        llm_calls = self.timing["llm_calls"]
        assert isinstance(llm_calls, list)
        llm_calls.append(
            {
                "phase": "answer",
                "provider": provider,
                "model": model,
                "endpoint": endpoint,
                "duration_s": round(duration, 3),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "prompt_preview": prompt[:120].replace("\n", " "),
            }
        )
        return result


class StaticDemoRetriever:
    def __init__(self) -> None:
        self._rows = [
            RetrievalResult(
                chunk=BaseChunk(
                    id="c-001",
                    text=(
                        "All electrical equipment must include grounding and emergency stop controls. "
                        "Periodic inspection interval is every 6 months."
                    ),
                    metadata={
                        "source_type": "normative",
                        "document_id": "doc-safety-01",
                        "version_id": "v3",
                        "page": "12",
                        "clause_path": "5.2.3",
                        "valid_to": "2099-12-31",
                    },
                    source_id="doc-safety-01",
                    chunk_index=0,
                ),
                score=0.94,
                provider="static-demo",
            ),
            RetrievalResult(
                chunk=BaseChunk(
                    id="c-002",
                    text=(
                        "Operator training is mandatory before commissioning. "
                        "A signed checklist must be stored in the project archive."
                    ),
                    metadata={
                        "source_type": "process",
                        "document_id": "doc-safety-02",
                        "version_id": "v1",
                        "page": "4",
                        "clause_path": "2.1",
                        "valid_to": "2099-12-31",
                    },
                    source_id="doc-safety-02",
                    chunk_index=0,
                ),
                score=0.89,
                provider="static-demo",
            ),
        ]

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        _ = question, filters
        return self._rows[:top_k]


def build_pipeline() -> TracingEnterprisePipeline:
    config = EnterprisePipelineConfig(
        intent_llm=EnterpriseIntentLLMConfig(
            provider="openai",
            model="Qwen/Qwen3-32B-AWQ",
            openai_base_url="http://hppii:8000/v1",
            openai_api_key="EMPTY",
            temperature=0.0,
            max_tokens=180,
            enable_tools=False,
            allowed_tools=["retrieve_context", "group_evidence", "score_risks", "build_attribution"],
            max_tool_calls=3,
        ),
        answer_llm=EnterpriseAnswerLLMConfig(
            provider="openai",
            model="Qwen/Qwen3-32B-AWQ",
            openai_base_url="http://hppii:8000/v1",
            openai_api_key="EMPTY",
            temperature=0.1,
            max_tokens=900,
            enable_tools=False,
            allowed_tools=[
                "retrieve_context",
                "group_evidence",
                "score_risks",
                "build_attribution",
                "create_word_document",
                "create_xlsx_table",
            ],
            max_tool_calls=5,
        ),
    )
    return TracingEnterprisePipeline(query_retriever=StaticDemoRetriever(), config=config)


def main() -> None:
    pipeline = build_pipeline()

    questions = [
        "Выдели ключевые требования безопасности и дай краткое резюме.",
        "Найди потенциальные риски несоответствия и предложи приоритетные действия.",
    ]

    results = []
    prompt_sources: list[str] = []
    timings: list[dict[str, object]] = []
    for idx, question in enumerate(questions, start=1):
        req = ReviewBundleRequest(
            session_id=f"vllm-demo-{idx}",
            query_text=question,
            document_ids=["doc-safety-01", "doc-safety-02"],
            department_principals=["safety-team", "qa-team"],
        )
        result = pipeline.review_bundle(req)
        results.append(result)
        prompt_sources.append(pipeline.last_prompt_source)
        timings.append(dict(pipeline.timing))

    output_dir = Path("artifacts")
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []
    for idx, item in enumerate(results, start=1):
        report_lines.append(f"Question {idx}: {questions[idx - 1]}")
        report_lines.append("Answer:")
        report_lines.append(item.llm_answer)
        report_lines.append("")

    word_res = pipeline.answer_tools.call(
        name="create_word_document",
        context=ToolContext(request_id="vllm-demo-artifacts"),
        args={
            "filename": str(output_dir / "enterprise_review.docx"),
            "title": "Enterprise Review (vLLM Demo)",
            "paragraphs": report_lines,
        },
    )

    xlsx_rows: list[list[str]] = []
    for idx, item in enumerate(results, start=1):
        xlsx_rows.append(
            [
                str(idx),
                questions[idx - 1],
                " | ".join(task.value for task in item.executed_tasks),
                item.llm_answer[:500],
            ]
        )

    xlsx_res = pipeline.answer_tools.call(
        name="create_xlsx_table",
        context=ToolContext(request_id="vllm-demo-artifacts"),
        args={
            "filename": str(output_dir / "enterprise_review.xlsx"),
            "sheet_name": "Answers",
            "headers": ["QuestionNo", "Question", "ExecutedTasks", "AnswerPreview"],
            "rows": xlsx_rows,
        },
    )

    print("Processed questions:")
    for idx, q in enumerate(questions, start=1):
        print(f"{idx}. {q}")

    print("\nPrompt diagnostics:")
    for idx, item in enumerate(results, start=1):
        source = prompt_sources[idx - 1]
        timing = timings[idx - 1]
        print(f"Question {idx}: uses PromptOrchestrator = {source == 'prompt_orchestrator'}")
        print(f"Prompt source: {source}")
        print(
            "Timing: "
            f"review_bundle={float(timing['review_bundle_time_s']):.3f}s, "
            f"llm_total={float(timing['llm_time_total_s']):.3f}s, "
            f"llm_calls={int(timing['llm_calls_total'])}, "
            f"intent={float(timing['intent_time_s']):.3f}s/{int(timing['intent_calls'])} calls, "
            f"planner={float(timing['planner_time_s']):.3f}s/{int(timing['planner_calls'])} calls, "
            f"answer={float(timing['answer_time_s']):.3f}s/{int(timing['answer_calls'])} calls, "
            f"tokens={int(timing['tokens_total'])} "
            f"(prompt={int(timing['prompt_tokens_total'])}, completion={int(timing['completion_tokens_total'])})"
        )
        print("LLM calls:")
        llm_calls = timing["llm_calls"]
        assert isinstance(llm_calls, list)
        for call in llm_calls:
            if not isinstance(call, dict):
                continue
            print(
                f"  - phase={call.get('phase')} provider={call.get('provider')} "
                f"model={call.get('model')} endpoint={call.get('endpoint')} "
                f"duration={call.get('duration_s')}s tokens={call.get('total_tokens')} "
                f"(prompt={call.get('prompt_tokens')}, completion={call.get('completion_tokens')})"
            )
        print("Generated prompt:")
        print(f"===== PROMPT_BEGIN_Q{idx} =====")
        print(item.prompt)
        print(f"===== PROMPT_END_Q{idx} =====")
        print("-" * 100)

    total_review = sum(float(t["review_bundle_time_s"]) for t in timings)
    total_llm = sum(float(t["llm_time_total_s"]) for t in timings)
    total_calls = sum(int(t["llm_calls_total"]) for t in timings)
    total_prompt_tokens = sum(int(t["prompt_tokens_total"]) for t in timings)
    total_completion_tokens = sum(int(t["completion_tokens_total"]) for t in timings)
    total_tokens = sum(int(t["tokens_total"]) for t in timings)
    print("\nTiming summary:")
    print(f"Total review_bundle time: {total_review:.3f}s")
    print(f"Total LLM time: {total_llm:.3f}s")
    print(f"Total LLM calls: {total_calls}")
    print(
        f"Total tokens: {total_tokens} "
        f"(prompt={total_prompt_tokens}, completion={total_completion_tokens})"
    )

    print("\nArtifacts:")
    print(f"DOCX -> {word_res}")
    print(f"XLSX -> {xlsx_res}")


if __name__ == "__main__":
    main()
