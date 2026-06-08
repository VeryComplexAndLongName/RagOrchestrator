from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

from ragflow_orchestrator.config.module_config import ModuleConfig
from ragflow_orchestrator.orchestrator import IngestSummary


class LanguageMode(str, Enum):
    """Language tagging strategy applied to ingested content."""

    AUTO = "auto"
    FORCE_RU = "force_ru"
    FORCE_EN = "force_en"
    MIXED = "mixed"


class IngestionError(BaseModel):
    source: str = Field(description="Identifier or path of the source item that failed or was skipped.")
    reason: str = Field(description="Human-readable explanation of the failure or skip reason.")


class TemplateRunMetrics(BaseModel):
    total_duration_ms: float = Field(default=0.0, description="Total wall-clock duration of the template run in milliseconds.")
    total_chunks: int = Field(default=0, description="Number of chunks successfully ingested across all sources.")
    duplicate_chunks_skipped: int = Field(default=0, description="Number of chunks skipped because they were already present.")
    chunks_per_second: float = Field(default=0.0, description="Ingestion throughput calculated as total_chunks divided by duration.")


class TemplateQualityMetric(BaseModel):
    strategy_name: str = Field(description="Name of the retrieval or ranking strategy being evaluated.")
    precision_at_k: float = Field(description="Precision@K score from the evaluation dataset.")
    recall_at_k: float = Field(description="Recall@K score from the evaluation dataset.")
    mrr: float = Field(description="Mean reciprocal rank of the first relevant result.")
    ndcg_at_k: float = Field(description="Normalized discounted cumulative gain at K.")


class TemplateRunReport(BaseModel):
    ingested: list[IngestSummary] = Field(
        default_factory=list,
        description="Summaries of sources that were successfully ingested.",
    )
    skipped: list[IngestionError] = Field(
        default_factory=list,
        description="Sources that were skipped without raising an exception.",
    )
    failed: list[IngestionError] = Field(
        default_factory=list,
        description="Sources that failed during ingestion with an error.",
    )
    run_metrics: TemplateRunMetrics | None = Field(
        default=None,
        description="Optional runtime performance metrics for the template run.",
    )
    quality: list[TemplateQualityMetric] = Field(
        default_factory=list,
        description="Optional retrieval quality metrics computed after ingestion.",
    )


class TemplateEvaluationConfig(BaseModel):
    enabled: bool = Field(default=False, description="Whether to run post-ingestion retrieval evaluation.")
    dataset_path: str = Field(
        default="datasets/retrieval_eval.jsonl",
        description="Path to the JSONL file with labeled queries and expected chunks.",
    )
    top_k: int = Field(default=3, description="Number of top results to consider when computing quality metrics.")


class TemplateExperimentLogConfig(BaseModel):
    enabled: bool = Field(default=True, description="Whether to persist template run results to the experiment journal.")
    db_path: str = Field(default="loadtest/experiments.sqlite", description="SQLite database path for experiment logging.")


class WebCrawlConfig(BaseModel):
    urls: list[str] = Field(description="Seed URLs to start crawling from.")
    max_depth: int = Field(default=1, description="Maximum link depth to follow from each seed URL.")
    same_domain_only: bool = Field(default=True, description="Restrict crawling to the same domain as the seed URL.")
    max_pages: int = Field(default=200, description="Maximum number of pages to fetch across all seeds.")
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for crawled page content.",
    )


class DocumentFolderConfig(BaseModel):
    folders: list[str] = Field(description="Local folder paths to scan for documents.")
    recursive: bool = Field(default=True, description="Whether to scan subdirectories recursively.")
    extensions: list[str] = Field(
        default_factory=lambda: [".docx", ".pdf", ".xlsx", ".txt", ".md", ".html"],
        description="File extensions to include during folder scanning.",
    )
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for extracted document text.",
    )


class RepoCodeConfig(BaseModel):
    repos: list[str] = Field(description="Local repository root paths to scan for source files.")
    recursive: bool = Field(default=True, description="Whether to scan repository subdirectories recursively.")
    include_hidden: bool = Field(default=False, description="Whether to include hidden files and directories.")
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
        ],
        description="Source file extensions to ingest from repositories.",
    )
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for ingested source code and docs.",
    )


class EmailTicketConfig(BaseModel):
    sources: list[str] = Field(description="File or folder paths containing support ticket exports.")
    recursive: bool = Field(default=True, description="Whether to scan subdirectories when a source is a folder.")
    extensions: list[str] = Field(
        default_factory=lambda: [".eml", ".jsonl", ".csv", ".txt", ".md"],
        description="Ticket file extensions to include during scanning.",
    )
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for ticket message content.",
    )


class IncrementalSyncConfig(BaseModel):
    folders: list[str] = Field(description="Local folder paths monitored for new or changed files.")
    recursive: bool = Field(default=True, description="Whether to monitor subdirectories recursively.")
    extensions: list[str] = Field(
        default_factory=lambda: [".docx", ".pdf", ".xlsx", ".txt", ".md", ".html"],
        description="File extensions eligible for incremental ingestion.",
    )
    state_file: str = Field(
        default=".rag_incremental_state.json",
        description="Path to the JSON file storing last-seen file hashes and timestamps.",
    )
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for newly synced content.",
    )


class ConfluenceWikiConfig(BaseModel):
    base_url: str = Field(description="Base URL of the Confluence instance (e.g. https://wiki.example.com).")
    page_ids: list[str] = Field(
        default_factory=list,
        description="Explicit Confluence page IDs to ingest.",
    )
    space_keys: list[str] = Field(
        default_factory=list,
        description="Confluence space keys whose pages should be ingested.",
    )
    max_pages: int = Field(default=200, description="Maximum number of pages to fetch per run.")
    auth_mode: str = Field(
        default="none",
        description="Authentication mode: none, bearer, or basic.",
    )
    username: str | None = Field(default=None, description="Username for basic authentication.")
    password: str | None = Field(default=None, description="Password for basic authentication.")
    token: str | None = Field(default=None, description="Bearer or personal access token.")
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for Confluence page content.",
    )


class JiraConfig(BaseModel):
    base_url: str = Field(description="Base URL of the Jira instance (e.g. https://jira.example.com).")
    jql: str = Field(default="order by updated desc", description="JQL query used to select issues for ingestion.")
    max_issues: int = Field(default=200, description="Maximum number of issues to fetch per run.")
    include_comments: bool = Field(default=True, description="Whether to include issue comments in ingested text.")
    auth_mode: str = Field(
        default="none",
        description="Authentication mode: none, bearer, or basic.",
    )
    username: str | None = Field(default=None, description="Username for basic authentication.")
    password: str | None = Field(default=None, description="Password for basic authentication.")
    token: str | None = Field(default=None, description="Bearer or personal access token.")
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for Jira issue content.",
    )


class APIReferenceConfig(BaseModel):
    sources: list[str] = Field(description="Local file paths or URLs pointing to OpenAPI/Swagger specifications.")
    include_operations: bool = Field(default=True, description="Whether to ingest API path and operation descriptions.")
    include_schemas: bool = Field(default=True, description="Whether to ingest component schema definitions.")
    max_items: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap for JSON array payload items to ingest.",
    )
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for API reference text.",
    )


class PyPIConfig(BaseModel):
    packages: list[str] = Field(
        default_factory=list,
        description="PyPI package names to ingest.",
    )
    include_release_history: bool = Field(default=True, description="Whether to include release version history.")
    max_releases_per_package: int = Field(
        default=20,
        description="Maximum number of recent releases to include per package.",
    )
    include_project_urls: bool = Field(default=True, description="Whether to include project homepage and doc URLs.")
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for package metadata text.",
    )


class GitHubConfig(BaseModel):
    owners: list[str] = Field(
        default_factory=list,
        description="GitHub usernames or organization names whose repositories should be ingested.",
    )
    max_projects: int = Field(default=50, description="Maximum total number of repositories to ingest per run.")
    max_repos_per_owner: int = Field(default=20, description="Maximum repositories to fetch per owner.")
    include_readme: bool = Field(default=True, description="Whether to fetch and include each repository README.")
    include_contributors: bool = Field(
        default=True,
        description="Whether to fetch contributors and persist them in the repository graph.",
    )
    auth_mode: str = Field(default="none", description="Authentication mode: none or bearer.")
    token: str | None = Field(default=None, description="GitHub personal access token for bearer authentication.")
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for repository content.",
    )


class GitLabConfig(BaseModel):
    base_url: str = Field(default="https://gitlab.com", description="Base URL of the GitLab instance.")
    groups_or_users: list[str] = Field(
        default_factory=list,
        description="GitLab group paths or usernames whose projects should be ingested.",
    )
    max_projects: int = Field(default=50, description="Maximum total number of projects to ingest per run.")
    max_repos_per_owner: int = Field(default=20, description="Maximum projects to fetch per group or user.")
    include_readme: bool = Field(default=True, description="Whether to fetch and include each project README.")
    include_contributors: bool = Field(
        default=True,
        description="Whether to fetch contributors and persist them in the repository graph.",
    )
    auth_mode: str = Field(default="none", description="Authentication mode: none or bearer.")
    token: str | None = Field(default=None, description="GitLab personal access token for bearer authentication.")
    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for project content.",
    )


class GraphStoreConfig(BaseModel):
    db_path: str = Field(
        default=".rag_graph.sqlite",
        description="SQLite database path for the repository and contributor graph store.",
    )


class TemplatesConfig(BaseModel):
    orchestrator: ModuleConfig = Field(
        default_factory=ModuleConfig,
        description="RAG orchestrator provider, embedding, and pipeline configuration.",
    )
    graph_store: GraphStoreConfig = Field(
        default_factory=GraphStoreConfig,
        description="Configuration for the optional repository graph store.",
    )
    evaluation: TemplateEvaluationConfig = Field(
        default_factory=TemplateEvaluationConfig,
        description="Post-ingestion retrieval evaluation settings.",
    )
    experiment_log: TemplateExperimentLogConfig = Field(
        default_factory=TemplateExperimentLogConfig,
        description="Experiment journal persistence settings.",
    )
    active_scenario: str = Field(
        default="document_folder",
        description="Name of the scenario to run from the scenarios map.",
    )
    scenarios: dict[str, dict] = Field(
        default_factory=dict,
        description="Map of scenario names to their template-specific configuration objects.",
    )


class BitrixConfig(BaseModel):
    domain: str = Field(description="Bitrix24 portal hostname without scheme (e.g. example.bitrix24.ru).")
    user_id: int = Field(ge=1, description="Webhook user ID used in the REST URL path.")
    token: SecretStr = Field(description="Inbound webhook user token.")

    language_mode: LanguageMode = Field(
        default=LanguageMode.AUTO,
        description="Language tagging strategy for Bitrix24 entity content.",
    )

    include_contacts: bool = Field(default=True, description="Whether to ingest CRM contacts.")
    include_companies: bool = Field(default=True, description="Whether to ingest CRM companies.")
    include_deals: bool = Field(default=True, description="Whether to ingest CRM deals.")
    include_leads: bool = Field(default=True, description="Whether to ingest CRM leads.")
    include_tasks: bool = Field(default=True, description="Whether to ingest tasks.")
    include_activities: bool = Field(default=True, description="Whether to ingest CRM activities.")
    include_im_dialogs: bool = Field(default=False, description="Whether to ingest instant messaging dialog messages.")

    max_contacts: int = Field(default=1000, ge=1, description="Maximum number of contacts to fetch.")
    max_companies: int = Field(default=1000, ge=1, description="Maximum number of companies to fetch.")
    max_deals: int = Field(default=1000, ge=1, description="Maximum number of deals to fetch.")
    max_leads: int = Field(default=1000, ge=1, description="Maximum number of leads to fetch.")
    max_tasks: int = Field(default=1000, ge=1, description="Maximum number of tasks to fetch.")
    max_activities: int = Field(default=1000, ge=1, description="Maximum number of activities to fetch.")
    max_dialog_messages: int = Field(default=200, ge=1, description="Maximum number of IM messages to fetch per dialog.")

    dialog_ids: list[str] = Field(
        default_factory=list,
        description="IM dialog IDs to ingest when include_im_dialogs is enabled.",
    )

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
