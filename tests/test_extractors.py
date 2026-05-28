from __future__ import annotations

from rag_orchestrator.templates.extractors import extract_text


def test_extract_text_decodes_cp1251_txt(tmp_path) -> None:
    text = "Автопубликация через GitHub Actions и release tag"
    path = tmp_path / "cp1251_note.txt"
    path.write_bytes(text.encode("cp1251"))

    extracted = extract_text(path)

    assert "GitHub Actions" in extracted
    assert "release tag" in extracted
    assert "Автопубликация" in extracted
    assert "�" not in extracted
