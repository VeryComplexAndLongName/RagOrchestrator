from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ragflow_orchestrator.config.module_config import ModuleConfig
from ragflow_orchestrator.orchestrator import IngestSummary


class LanguageMode(str, Enum):
    AUTO = "auto"
    FORCE_RU = "force_ru"
    FORCE_EN = "force_en"
    MIXED = "mixed"


class IngestionError(BaseModel):
    source: str
    reason: str


class TemplateRunMetrics(BaseModel):
    total_duration_ms: float = 0.0
    total_chunks: int = 0
    duplicate_chunks_skipped: int = 0
    chunks_per_second: float = 0.0


class TemplateQualityMetric(BaseModel):
    strategy_name: str
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float


class TemplateRunReport(BaseModel):
    ingested: list[IngestSummary] = Field(default_factory=list)
    skipped: list[IngestionError] = Field(default_factory=list)
    failed: list[IngestionError] = Field(default_factory=list)
    run_metrics: TemplateRunMetrics | None = None
    quality: list[TemplateQualityMetric] = Field(default_factory=list)


class TemplateEvaluationConfig(BaseModel):
    enabled: bool = False
    dataset_path: str = "datasets/retrieval_eval.jsonl"
    top_k: int = 3


class TemplateExperimentLogConfig(BaseModel):
    enabled: bool = True
    db_path: str = "loadtest/experiments.sqlite"


class WebCrawlConfig(BaseModel):
    urls: list[str]
    max_depth: int = 1
    same_domain_only: bool = True
    max_pages: int = 200
    language_mode: LanguageMode = LanguageMode.AUTO


class DocumentFolderConfig(BaseModel):
    folders: list[str]
    recursive: bool = True
    extensions: list[str] = Field(default_factory=lambda: [".docx", ".pdf", ".xlsx", ".txt", ".md", ".html"])
    language_mode: LanguageMode = LanguageMode.AUTO


class RepoCodeConfig(BaseModel):
    repos: list[str]
    recursive: bool = True
    include_hidden: bool = False
    extensions: list[str] = Field(
        default_factory=lambda: [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".go",
            ".java",
            ".cs",
            ".md",
            ".yaml",
            ".yml",
            ".json",
        ]
    )
    language_mode: LanguageMode = LanguageMode.AUTO


class EmailTicketConfig(BaseModel):
    sources: list[str]
    recursive: bool = True
    extensions: list[str] = Field(default_factory=lambda: [".eml", ".jsonl", ".csv", ".txt", ".md"])
    language_mode: LanguageMode = LanguageMode.AUTO


class IncrementalSyncConfig(BaseModel):
    folders: list[str]
    recursive: bool = True
    extensions: list[str] = Field(default_factory=lambda: [".docx", ".pdf", ".xlsx", ".txt", ".md", ".html"])
    state_file: str = ".rag_incremental_state.json"
    language_mode: LanguageMode = LanguageMode.AUTO


class ConfluenceWikiConfig(BaseModel):
    base_url: str
    page_ids: list[str] = Field(default_factory=list)
    space_keys: list[str] = Field(default_factory=list)
    max_pages: int = 200
    auth_mode: str = "none"  # none | bearer | basic
    username: str | None = None
    password: str | None = None
    token: str | None = None
    language_mode: LanguageMode = LanguageMode.AUTO


class JiraConfig(BaseModel):
    base_url: str
    jql: str = "order by updated desc"
    max_issues: int = 200
    include_comments: bool = True
    auth_mode: str = "none"  # none | bearer | basic
    username: str | None = None
    password: str | None = None
    token: str | None = None
    language_mode: LanguageMode = LanguageMode.AUTO


class APIReferenceConfig(BaseModel):
    sources: list[str]
    include_operations: bool = True
    include_schemas: bool = True
    language_mode: LanguageMode = LanguageMode.AUTO


class PyPIConfig(BaseModel):
    packages: list[str] = Field(default_factory=list)
    include_release_history: bool = True
    max_releases_per_package: int = 20
    include_project_urls: bool = True
    language_mode: LanguageMode = LanguageMode.AUTO


class GitHubConfig(BaseModel):
    owners: list[str] = Field(default_factory=list)
    max_projects: int = 50
    max_repos_per_owner: int = 20
    include_readme: bool = True
    include_contributors: bool = True
    auth_mode: str = "none"  # none | bearer
    token: str | None = None
    language_mode: LanguageMode = LanguageMode.AUTO


class GitLabConfig(BaseModel):
    base_url: str = "https://gitlab.com"
    groups_or_users: list[str] = Field(default_factory=list)
    max_projects: int = 50
    max_repos_per_owner: int = 20
    include_readme: bool = True
    include_contributors: bool = True
    auth_mode: str = "none"  # none | bearer
    token: str | None = None
    language_mode: LanguageMode = LanguageMode.AUTO


class GraphStoreConfig(BaseModel):
    db_path: str = ".rag_graph.sqlite"


class TemplatesConfig(BaseModel):
    orchestrator: ModuleConfig = Field(default_factory=ModuleConfig)
    graph_store: GraphStoreConfig = Field(default_factory=GraphStoreConfig)
    evaluation: TemplateEvaluationConfig = Field(default_factory=TemplateEvaluationConfig)
    experiment_log: TemplateExperimentLogConfig = Field(default_factory=TemplateExperimentLogConfig)
    active_scenario: str = "document_folder"
    scenarios: dict[str, dict] = Field(default_factory=dict)
