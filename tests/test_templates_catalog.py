from __future__ import annotations

from importlib import import_module

from ragflow_orchestrator.templates.catalog import list_installed_templates


def test_list_installed_templates_contains_name_and_description() -> None:
    items = list_installed_templates()

    assert items
    assert all(item["name"] for item in items)
    assert all(item["description"] for item in items)

    names = {item["name"] for item in items}
    assert "document_folder" in names
    assert "web_crawl" in names
    assert "github" in names


def test_list_installed_templates_skips_unimportable_modules(monkeypatch) -> None:
    def fake_import_module(module_name: str):
        if module_name == "ragflow_orchestrator.templates.github_template":
            raise ModuleNotFoundError("github template unavailable")
        return import_module(module_name)

    monkeypatch.setattr("ragflow_orchestrator.templates.catalog.import_module", fake_import_module)

    names = {item["name"] for item in list_installed_templates()}
    assert "github" not in names
    assert "document_folder" in names
