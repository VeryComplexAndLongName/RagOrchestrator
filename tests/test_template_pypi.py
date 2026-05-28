from __future__ import annotations

import json
from pathlib import Path

from rag_orchestrator.embedding import HashEmbedder
from rag_orchestrator.factory import create_provider
from rag_orchestrator.orchestrator import RAGOrchestrator
from rag_orchestrator.presets import document_preset
from rag_orchestrator.templates import LanguageMode, PyPIConfig, PyPITemplate
from rag_orchestrator.templates.runner import run_template_from_json


def _build_orchestrator(tmp_path: Path) -> RAGOrchestrator:
    provider = create_provider("sqlite+vec", db_path=str(tmp_path / "pypi.db"), table_name="pypi_chunks")
    preset = document_preset()
    return RAGOrchestrator(
        provider=provider,
        embedder=HashEmbedder(dimensions=64),
        chunker=preset.chunker,
        cleaner=preset.cleaner,
    )


def test_pypi_template_with_stub_payload(tmp_path: Path) -> None:
    orchestrator = _build_orchestrator(tmp_path)
    template = PyPITemplate(orchestrator)

    template._fetch_package_payload = lambda package: {  # type: ignore[method-assign]
        "info": {
            "name": package,
            "version": "1.0.0",
            "summary": "Demo package",
            "description": "Long package description",
            "home_page": "https://example.com",
            "license": "MIT",
            "requires_python": ">=3.11",
            "classifiers": ["Programming Language :: Python :: 3"],
            "project_urls": {"Source": "https://example.com/repo"},
        },
        "releases": {
            "1.0.0": [
                {
                    "filename": "demo-1.0.0.tar.gz",
                    "packagetype": "sdist",
                    "python_version": "source",
                    "size": 1024,
                    "upload_time": "2026-01-01T00:00:00",
                }
            ]
        },
    }

    report = template.run(
        PyPIConfig(
            packages=["demo-package"],
            include_release_history=True,
            max_releases_per_package=5,
            include_project_urls=True,
            language_mode=LanguageMode.AUTO,
        )
    )

    assert report.ingested
    assert not report.failed


def test_pypi_build_chunks_includes_dependencies_and_authors() -> None:
    payload = {
        "info": {
            "name": "demo-package",
            "version": "1.2.3",
            "summary": "Demo package",
            "description": "<p>HTML description</p>",
            "home_page": "https://example.com",
            "package_url": "https://pypi.org/project/demo-package/",
            "project_url": "https://pypi.org/project/demo-package/",
            "docs_url": "https://docs.example.com",
            "download_url": "",
            "bugtrack_url": "https://example.com/issues",
            "license": "MIT",
            "requires_python": ">=3.11",
            "author": "Jane Dev",
            "author_email": "jane@example.com",
            "maintainer": "Ops Team",
            "maintainer_email": "ops@example.com",
            "keywords": "rag,orchestrator",
            "classifiers": ["Programming Language :: Python :: 3"],
            "project_urls": {"Source": "https://example.com/repo"},
            "requires_dist": ["pydantic>=2", "httpx>=0.27"],
            "provides_extra": ["dev"],
        },
        "releases": {},
    }

    chunks = PyPITemplate._build_chunks(
        package="demo-package",
        payload=payload,
        include_release_history=False,
        max_releases_per_package=5,
        include_project_urls=True,
    )

    full_text = "\n".join(chunks)
    assert "Author: Jane Dev" in full_text
    assert "Maintainer: Ops Team" in full_text
    assert "Dependencies (requires_dist):" in full_text
    assert "- pydantic>=2" in full_text
    assert "- httpx>=0.27" in full_text


def test_pypi_build_chunks_has_short_fact_url_chunk_and_plain_markdown() -> None:
    payload = {
        "info": {
            "name": "fastapi",
            "version": "1.0.0",
            "summary": "Fast API",
            "description": "**Source Code**: [https://github.com/fastapi/fastapi](https://github.com/fastapi/fastapi)",
            "home_page": "https://fastapi.tiangolo.com",
            "package_url": "https://pypi.org/project/fastapi/",
            "project_url": "https://pypi.org/project/fastapi/",
            "docs_url": "https://fastapi.tiangolo.com",
            "download_url": "",
            "bugtrack_url": "",
            "license": "MIT",
            "requires_python": ">=3.9",
            "classifiers": [],
            "project_urls": {
                "Source": "https://github.com/fastapi/fastapi",
            },
        },
        "releases": {},
    }

    chunks = PyPITemplate._build_chunks(
        package="fastapi",
        payload=payload,
        include_release_history=False,
        max_releases_per_package=5,
        include_project_urls=True,
    )

    assert chunks
    first_chunk = chunks[0]
    assert "Key URLs:" in first_chunk
    assert "Repository: https://github.com/fastapi/fastapi" in first_chunk

    full_text = "\n".join(chunks)
    assert "Source Code: https://github.com/fastapi/fastapi" in full_text
    assert "**Source Code**" not in full_text


def test_run_template_from_json_with_pypi(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "rag_orchestrator.templates.pypi.PyPITemplate._fetch_package_payload",
        lambda self, package: {
            "info": {
                "name": package,
                "version": "0.0.1",
                "summary": "Pkg",
                "description": "Desc",
                "home_page": "",
                "license": "",
                "requires_python": ">=3.11",
                "classifiers": [],
                "project_urls": {},
            },
            "releases": {},
        },
    )

    cfg = {
        "orchestrator": {
            "provider": {
                "kind": "sqlite+vec",
                "params": {
                    "db_path": str(tmp_path / "runner_pypi.db"),
                    "table_name": "runner_pypi_chunks",
                },
            },
            "embedding": {
                "provider": "hash",
                "options": {"dimensions": 64},
            },
            "pipeline": {"preset": "document"},
        },
        "active_scenario": "pypi",
        "scenarios": {
            "pypi": {
                "packages": ["requests"],
                "include_release_history": True,
                "max_releases_per_package": 5,
                "include_project_urls": True,
                "language_mode": "auto",
            }
        },
    }

    cfg_path = tmp_path / "templates.json"
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=True, indent=2), encoding="utf-8")

    report = run_template_from_json(str(cfg_path))
    assert report.ingested
    assert not report.failed
