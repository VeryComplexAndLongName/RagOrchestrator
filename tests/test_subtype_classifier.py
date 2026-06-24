from __future__ import annotations

from ragflow_orchestrator.config import SubtypeClassificationConfig, SubtypeLLMConfig
from ragflow_orchestrator.subtype_classifier import DocumentSubtypeClassifier


def test_rules_detect_normative() -> None:
    classifier = DocumentSubtypeClassifier()
    text = (
        "Раздел 1 Общие положения\n"
        "1.1 Требования к надежности\n"
        "1.2 Пункт по безопасности\n"
        "См. ГОСТ 27751-2014 и СП 14.13330.2018"
    )

    result = classifier.predict(text=text, title="Нормативный документ", document_type="pdf")

    assert result.subtype == "normative"
    assert result.confidence > 0
    assert result.source in {"rules", "rules+llm"}


def test_rules_detect_agreement() -> None:
    classifier = DocumentSubtypeClassifier()
    text = (
        "Договор оказания услуг\n"
        "Стороны согласовали предмет договора и срок действия.\n"
        "Подписи сторон расположены в конце документа."
    )

    result = classifier.predict(text=text, title="Соглашение", document_type="docx")

    assert result.subtype == "agreement"


def test_fallback_unknown_on_low_confidence() -> None:
    config = SubtypeClassificationConfig(
        confidence_threshold=0.9,
        fallback_subtype="description",
        llm=SubtypeLLMConfig(enabled=False),
    )
    classifier = DocumentSubtypeClassifier(config=config)

    result = classifier.predict(text="x", title=None, document_type="txt")

    assert result.subtype in {"unknown", "description"}
    assert result.source == "fallback"


def test_disabled_classifier_returns_fallback() -> None:
    config = SubtypeClassificationConfig(enabled=False, fallback_subtype="description")
    classifier = DocumentSubtypeClassifier(config=config)

    result = classifier.predict(text="Any text", title="Any", document_type="txt")

    assert result.subtype == "description"
    assert result.source == "disabled"
    assert result.confidence == 1.0
