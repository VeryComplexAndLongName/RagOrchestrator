from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any
from urllib import error, request

from ragflow_orchestrator.config import SubtypeClassificationConfig


@dataclass(slots=True)
class SubtypePrediction:
    subtype: str
    confidence: float
    source: str
    rules_scores: dict[str, float]
    llm_scores: dict[str, float]


class DocumentSubtypeClassifier:
    """Hybrid document subtype classifier: rules + optional LLM + confidence fallback."""

    def __init__(self, config: SubtypeClassificationConfig | None = None) -> None:
        self.config = config or SubtypeClassificationConfig()
        self._allowed = set(self.config.allowed_subtypes)

    def predict(
        self,
        *,
        text: str,
        title: str | None = None,
        document_type: str | None = None,
    ) -> SubtypePrediction:
        if not self.config.enabled:
            return SubtypePrediction(
                subtype=self.config.fallback_subtype,
                confidence=1.0,
                source="disabled",
                rules_scores={},
                llm_scores={},
            )

        rules_scores = self._rules_score(text=text, title=title, document_type=document_type)
        llm_scores: dict[str, float] = {}

        if self.config.llm.enabled and self.config.llm.provider != "none":
            llm_scores = self._llm_score(text=text, title=title)

        merged = self._merge_scores(rules_scores, llm_scores)
        if not merged:
            return SubtypePrediction(
                subtype=self.config.fallback_subtype,
                confidence=0.0,
                source="fallback",
                rules_scores=rules_scores,
                llm_scores=llm_scores,
            )

        best_subtype, best_score = max(merged.items(), key=lambda kv: kv[1])
        if best_score < self.config.confidence_threshold:
            return SubtypePrediction(
                subtype="unknown" if "unknown" in self._allowed else self.config.fallback_subtype,
                confidence=best_score,
                source="fallback",
                rules_scores=rules_scores,
                llm_scores=llm_scores,
            )

        source = "rules+llm" if llm_scores else "rules"
        return SubtypePrediction(
            subtype=best_subtype,
            confidence=best_score,
            source=source,
            rules_scores=rules_scores,
            llm_scores=llm_scores,
        )

    def _rules_score(
        self,
        *,
        text: str,
        title: str | None,
        document_type: str | None,
    ) -> dict[str, float]:
        content = f"{title or ''}\n{text[:12000]}".lower()
        scores = {name: 0.0 for name in self._allowed}

        def add(name: str, weight: float, condition: bool) -> None:
            if condition and name in scores:
                scores[name] += weight

        # normative
        clause_hits = len(re.findall(r"\b\d+(?:\.\d+){1,4}\b", content))
        add("normative", min(0.5, clause_hits * 0.03), clause_hits > 0)
        add("normative", 0.35, bool(re.search(r"\b(раздел|пункт|статья|гост|снип|сп\s+\d)", content)))
        add("normative", 0.2, clause_hits >= 2)
        add("normative", 0.2, bool(re.search(r"\b(норматив|standard|regulation)\b", content)))

        # agreement
        add("agreement", 0.35, bool(re.search(r"\b(agreement|договор|соглашение|стороны)\b", content)))
        add("agreement", 0.25, bool(re.search(r"\b(предмет\s+договора|срок\s+действия|подписи\s+сторон)\b", content)))
        add("agreement", 0.2, bool(re.search(r"\b(договора|соглашени[ея])\b", content)) and bool(re.search(r"\bсторон[аы]?\b", content)))

        # contract legal
        add("contract_legal", 0.35, bool(re.search(r"\b(обязуется|ответственность|юридическ|штраф|неустойк)\b", content)))
        add("contract_legal", 0.2, bool(re.search(r"\b(истец|ответчик|арбитраж|подсудност)\b", content)))

        # instruction
        add("instruction", 0.35, bool(re.search(r"\b(шаг\s*\d+|порядок\s+действий|инструкция|выполните|нажмите)\b", content)))
        add("instruction", 0.2, bool(re.search(r"\b(должен\s+быть\s+выполнен|последовательно)\b", content)))

        # policy
        add("policy", 0.35, bool(re.search(r"\b(политика|регламент|правила|compliance|governance)\b", content)))

        # specification
        add("specification", 0.3, bool(re.search(r"\b(требовани[ея]|характеристик|параметр|допуск|спецификац)\b", content)))
        add("specification", 0.15, bool(re.search(r"\b(таблица|table|unit|размер)\b", content)))

        # report
        add("report", 0.35, bool(re.search(r"\b(отчет|report|итоги|результаты|метрики|выводы)\b", content)))

        # faq
        add("faq", 0.45, bool(re.search(r"\b(faq|q:\s|вопрос\s*[:\-]|ответ\s*[:\-])\b", content)))

        # reference
        add("reference", 0.3, bool(re.search(r"\b(справочник|reference|глоссарий|термин)\b", content)))

        # code_doc
        add("code_doc", 0.35, bool(re.search(r"\b(api|endpoint|method|параметр\s+запроса|response|sdk)\b", content)))

        # description
        add("description", 0.15, True)
        avg_line_len = self._avg_line_length(content)
        add("description", 0.25, avg_line_len > 45)
        add("description", 0.1, clause_hits < 3)

        # priors by container type
        if document_type:
            dt = document_type.lower()
            add("code_doc", 0.1, dt == "code")
            add("normative", 0.08, dt == "pdf")
            add("specification", 0.08, dt == "xlsx")

        # keep only allowed (raw confidence-like scores)
        filtered = {k: max(0.0, v) for k, v in scores.items() if k in self._allowed}
        return filtered

    def _llm_score(self, *, text: str, title: str | None) -> dict[str, float]:
        provider = self.config.llm.provider
        if provider == "ollama":
            return self._llm_score_ollama(text=text, title=title)
        if provider == "openai_compat":
            return self._llm_score_openai_compat(text=text, title=title)
        return {}

    def _llm_score_ollama(self, *, text: str, title: str | None) -> dict[str, float]:
        base_url = (self.config.llm.base_url or "http://localhost:11434").rstrip("/")
        model = self.config.llm.model
        if not model:
            return {}

        prompt = self._build_prompt(text=text, title=title)
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": self.config.llm.temperature},
        }
        raw = self._http_post_json(
            url=f"{base_url}/api/generate",
            payload=payload,
            timeout_seconds=self.config.llm.timeout_seconds,
            headers={},
        )
        if not raw:
            return {}
        response_text = str(raw.get("response") or "")
        return self._extract_llm_scores(response_text)

    def _llm_score_openai_compat(self, *, text: str, title: str | None) -> dict[str, float]:
        base_url = (self.config.llm.base_url or "https://api.openai.com").rstrip("/")
        model = self.config.llm.model
        if not model:
            return {}

        api_key = os.getenv(self.config.llm.api_key_env, "")
        if not api_key:
            return {}

        prompt = self._build_prompt(text=text, title=title)
        payload = {
            "model": model,
            "temperature": self.config.llm.temperature,
            "messages": [
                {"role": "system", "content": "You classify document subtype and return JSON only."},
                {"role": "user", "content": prompt},
            ],
        }
        raw = self._http_post_json(
            url=f"{base_url}/v1/chat/completions",
            payload=payload,
            timeout_seconds=self.config.llm.timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        if not raw:
            return {}
        choices = raw.get("choices") or []
        if not choices:
            return {}
        content = str(choices[0].get("message", {}).get("content", ""))
        return self._extract_llm_scores(content)

    def _extract_llm_scores(self, response_text: str) -> dict[str, float]:
        try:
            payload = json.loads(response_text)
        except json.JSONDecodeError:
            return {}

        subtype = str(payload.get("subtype") or "").strip().lower()
        confidence = payload.get("confidence", 0.0)
        try:
            confidence_value = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence_value = 0.0

        if subtype not in self._allowed:
            return {}
        return {subtype: confidence_value}

    def _build_prompt(self, *, text: str, title: str | None) -> str:
        labels = sorted(self._allowed)
        sample = text[:6000]
        return (
            "Classify the document into one subtype label. "
            "Return valid JSON object only: {\"subtype\": \"<label>\", \"confidence\": <0..1>}\n"
            f"Allowed labels: {', '.join(labels)}\n"
            f"Title: {title or ''}\n"
            f"Text:\n{sample}"
        )

    @staticmethod
    def _http_post_json(
        *,
        url: str,
        payload: dict[str, Any],
        timeout_seconds: float,
        headers: dict[str, str],
    ) -> dict[str, Any] | None:
        body = json.dumps(payload).encode("utf-8")
        req_headers = {"Content-Type": "application/json", **headers}
        req = request.Request(url, data=body, headers=req_headers, method="POST")
        try:
            with request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw)
        except (error.URLError, error.HTTPError, TimeoutError, json.JSONDecodeError):
            return None

    def _merge_scores(self, rules: dict[str, float], llm: dict[str, float]) -> dict[str, float]:
        labels = self._allowed
        out: dict[str, float] = {}
        rw = max(0.0, min(1.0, self.config.rules_weight))
        lw = max(0.0, min(1.0, self.config.llm_weight))

        if not llm:
            rw = 1.0
            lw = 0.0

        for label in labels:
            r = rules.get(label, 0.0)
            l = llm.get(label, 0.0)
            out[label] = max(0.0, min(1.0, (r * rw) + (l * lw)))

        return out

    @staticmethod
    def _normalize(scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        total = sum(max(0.0, v) for v in scores.values())
        if total <= 0:
            return {k: 0.0 for k in scores}
        return {k: max(0.0, v) / total for k, v in scores.items()}

    @staticmethod
    def _avg_line_length(text: str) -> float:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return 0.0
        return sum(len(line) for line in lines) / len(lines)
