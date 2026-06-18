from __future__ import annotations

from ragflow_orchestrator.boilerplate import (
    BoilerplateAggressiveness,
    BoilerplateRemovalResult,
    BoilerplateRemover,
)
from ragflow_orchestrator.document_pipeline import AdaptiveDocumentChunker


class _DropNoiseRemover:
    name = "drop-noise"

    def remove(
        self,
        text: str,
        *,
        document_type: str,
        aggressiveness: BoilerplateAggressiveness,
        metadata: dict[str, object] | None = None,
    ) -> BoilerplateRemovalResult:
        del document_type, aggressiveness, metadata
        lines = text.splitlines()
        kept = [line for line in lines if "NOISE" not in line]
        return BoilerplateRemovalResult(
            text="\n".join(kept).strip(),
            removed_lines=len(lines) - len(kept),
            total_lines=len(lines),
            remover=self.name,
            aggressiveness=BoilerplateAggressiveness.BALANCED,
        )


def test_txt_boilerplate_removed_with_balanced_aggressiveness() -> None:
    chunker = AdaptiveDocumentChunker()
    chunks = chunker.chunk(
        source_id="note.txt",
        text="Useful content\nPrivacy Policy\nAll rights reserved\nUseful ending",
        metadata={"document_type": "txt", "boilerplate_aggressiveness": "balanced"},
    )

    assert chunks
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "Privacy Policy" not in joined
    assert "All rights reserved" not in joined
    assert "Useful content" in joined
    # Check that metadata was propagated from raw text boilerplate removal to chunks.
    first_chunk = chunks[0]
    assert "boilerplate_aggressiveness" in first_chunk.metadata
    assert first_chunk.metadata["boilerplate_aggressiveness"] == "balanced"


def test_can_disable_boilerplate_removal_via_metadata_flag() -> None:
    chunker = AdaptiveDocumentChunker()
    chunks = chunker.chunk(
        source_id="note.txt",
        text="Useful content\nPrivacy Policy",
        metadata={"document_type": "txt", "boilerplate_remove": False},
    )

    assert chunks
    joined = "\n".join(chunk.text for chunk in chunks)
    assert "Privacy Policy" in joined


def test_can_register_and_set_custom_remover() -> None:
    chunker = AdaptiveDocumentChunker()
    custom: BoilerplateRemover = _DropNoiseRemover()
    chunker.register_boilerplate_remover(document_type="txt", name=custom.name, remover=custom, default=True)

    chunks = chunker.chunk(
        source_id="custom.txt",
        text="keep\nNOISE to remove\nkeep2",
        metadata={"document_type": "txt"},
    )

    assert chunks
    assert "NOISE" not in chunks[0].text
    assert chunks[0].metadata["boilerplate_remover"] == "drop-noise"


def test_html_can_select_justext_remover_by_name() -> None:
    chunker = AdaptiveDocumentChunker()
    chunks = chunker.chunk(
        source_id="page.html",
        text="<html><body><h1>Title</h1><p>Main content</p></body></html>",
        metadata={"document_type": "html", "boilerplate_remover": "justext"},
    )

    assert chunks
    assert chunks[0].metadata["boilerplate_remover"] == "justext"
