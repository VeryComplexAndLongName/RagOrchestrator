from __future__ import annotations
from enum import Enum

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

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

class BitrixConfig(BaseModel):
    domain: str                     # <domain>.bitrix24.ru
    user_id: int = Field(ge=1)      # ID webhook user
    token: SecretStr                # user token

    language_mode: LanguageMode = LanguageMode.AUTO

    include_contacts: bool = True
    include_companies: bool = True
    include_deals: bool = True
    include_leads: bool = True
    include_tasks: bool = True
    include_activities: bool = True
    include_im_dialogs: bool = False

    max_contacts: int = Field(default=1000, ge=1)
    max_companies: int = Field(default=1000, ge=1)
    max_deals: int = Field(default=1000, ge=1)
    max_leads: int = Field(default=1000, ge=1)
    max_tasks: int = Field(default=1000, ge=1)
    max_activities: int = Field(default=1000, ge=1)
    max_dialog_messages: int = Field(default=200, ge=1)

    dialog_ids: list[str] = Field(default_factory=list)

    @field_validator("domain")
    @classmethod
    def normalize_domain(cls, value: str) -> str:
        value = value.strip()
        value = value.removeprefix("https://")
        value = value.removeprefix("http://")
        value = value.rstrip("/")

        if not value:
            raise ValueError("domain cannot be empty")

        if any(ch in value for ch in ("/", "?", "#")):
            raise ValueError("domain must contain host only, without path/query/fragment")

        if " " in value:
            raise ValueError("domain cannot contain whitespace")

        return value

    @field_validator("token")
    @classmethod
    def validate_token(cls, value: SecretStr) -> SecretStr:
        token = value.get_secret_value().strip()
        if not token:
            raise ValueError("token cannot be empty")
        return SecretStr(token)

    @field_validator("dialog_ids", mode="before")
    @classmethod
    def normalize_dialog_ids(cls, value: object) -> list[str]:
        if value is None:
            return []

        if not isinstance(value, list):
            raise ValueError("dialog_ids must be a list of strings")

        normalized: list[str] = []
        seen: set[str] = set()
        for raw_item in value:
            item = str(raw_item).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)

        return normalized

    @model_validator(mode="after")
    def validate_im_dialog_dependency(self) -> "BitrixConfig":
        if self.include_im_dialogs and not self.dialog_ids:
            raise ValueError("dialog_ids must be provided when include_im_dialogs is true")
        return self

    @property
    def base_url(self) -> str:
        return (
            f"https://{self.domain}"
            f"/rest/{self.user_id}/{self.token.get_secret_value()}"
        )
    