from ragflow_orchestrator.cleaning import MarkupAwareTextCleaner
from ragflow_orchestrator.presets import document_preset, markdown_preset


def test_markup_aware_cleaner_strips_html_and_markdown_but_keeps_urls() -> None:
    raw = """
    <section>
      <h1>FastAPI</h1>
      <p>See [docs](https://fastapi.tiangolo.com) and <strong>install</strong> with `pip install fastapi`.</p>
      <script>window.bad = true;</script>
    </section>
    """

    cleaned = MarkupAwareTextCleaner().clean(raw)

    assert "<section>" not in cleaned
    assert "[docs](https://fastapi.tiangolo.com)" not in cleaned
    assert "docs: https://fastapi.tiangolo.com" in cleaned
    assert "pip install fastapi" in cleaned
    assert "window.bad = true;" not in cleaned
    assert "FastAPI" in cleaned


def test_document_and_markdown_presets_use_markup_aware_cleaner() -> None:
    assert type(document_preset().cleaner).__name__ == "MarkupAwareTextCleaner"
    assert type(markdown_preset().cleaner).__name__ == "MarkupAwareTextCleaner"


def test_markup_aware_cleaner_removes_common_markdown_markers() -> None:
  raw = """
  # Something
  ## Something else
  ### Third thing
  * Item one
  ---
  **Important**
  *Italic text*
  """

  cleaned = MarkupAwareTextCleaner().clean(raw)

  assert "# Something" not in cleaned
  assert "## Something else" not in cleaned
  assert "### Third thing" not in cleaned
  assert "* Item one" not in cleaned
  assert "---" not in cleaned
  assert "**Important**" not in cleaned
  assert "*Italic text*" not in cleaned

  assert "Something" in cleaned
  assert "Something else" in cleaned
  assert "Third thing" in cleaned
  assert "Item one" in cleaned
  assert "Important" in cleaned
  assert "Italic text" in cleaned


def test_markup_aware_cleaner_removes_indented_list_and_heading_markers() -> None:
  raw = "\n\t* Something\n    ## Heading\n      > quoted\n"

  cleaned = MarkupAwareTextCleaner().clean(raw)

  assert "* Something" not in cleaned
  assert "## Heading" not in cleaned
  assert "> quoted" not in cleaned
  assert "Something" in cleaned
  assert "Heading" in cleaned
  assert "quoted" in cleaned


def test_markup_aware_cleaner_normalizes_markdown_tables() -> None:
  raw = """
  | Ускоряет CPU |
  | --------------- | ------ | -------- | ------------ | ------------ |
  | asyncio | ❌ | ❌ | I/O | ❌ |
  | multithreading | ✅ | ❌ | I/O | ❌ (GIL) |
  | multiprocessing | ❌ | ✅ | CPU | ✅ |
  """

  cleaned = MarkupAwareTextCleaner().clean(raw)

  assert "|" not in cleaned
  assert "---------------" not in cleaned
  assert "asyncio; ❌; ❌; I/O; ❌" in cleaned
  assert "multithreading; ✅; ❌; I/O; ❌ (GIL)" in cleaned
  assert "multiprocessing; ❌; ✅; CPU; ✅" in cleaned