"""PostgreSQL schema migrations for document management with versioning and ACL.

This module provides migration functions (up/down) for PostgreSQL tables:
- documents (base documents)
- document_versions (versioned snapshots)
- chunks (text chunks linked to versions)
- Tags, ACL, path history, ingestion logs
- Views and functions for ACL authorization

Source of truth: MyTasks/sql.md
"""

from __future__ import annotations

from typing import Any

import psycopg  # psycopg 3


def create_postgres_connection(dsn: str) -> psycopg.Connection:
    """Create and return a PostgreSQL connection."""
    return psycopg.connect(dsn)


# ============================================================
# Migration Step 1: Core utility function (set_updated_at)
# ============================================================

def migration_001_create_trigger_function_up(conn: psycopg.Connection) -> None:
    """Create the set_updated_at trigger function."""
    sql = """
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

    CREATE OR REPLACE FUNCTION set_updated_at()
    RETURNS TRIGGER AS $$
    BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
    END;
    $$ LANGUAGE plpgsql;
    """
    conn.execute(sql)
    conn.commit()


def migration_001_create_trigger_function_down(conn: psycopg.Connection) -> None:
    """Drop the set_updated_at trigger function."""
    sql = "DROP FUNCTION IF EXISTS set_updated_at() CASCADE;"
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 2: Documents table
# ============================================================

def migration_002_create_documents_table_up(conn: psycopg.Connection) -> None:
    """Create documents table."""
    sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        source_type     TEXT NOT NULL,
        document_type   TEXT NOT NULL,
        title           TEXT,
        doc_number      TEXT,
        language        TEXT,
        domain          TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE TRIGGER trg_documents_updated
        BEFORE UPDATE ON documents
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();

    CREATE INDEX ix_documents_source_type ON documents(source_type);
    CREATE INDEX ix_documents_doc_number  ON documents(doc_number);
    """
    conn.execute(sql)
    conn.commit()


def migration_002_create_documents_table_down(conn: psycopg.Connection) -> None:
    """Drop documents table and related triggers."""
    sql = """
    DROP TABLE IF EXISTS documents CASCADE;
    """
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 3: Document versions table
# ============================================================

def migration_003_create_document_versions_table_up(conn: psycopg.Connection) -> None:
    """Create document_versions table."""
    sql = """
    CREATE TABLE IF NOT EXISTS document_versions (
        id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        document_id         UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        version_number      INTEGER NOT NULL,
        source_hash         TEXT,
        content_hash        TEXT NOT NULL,
        file_path           TEXT NOT NULL,
        file_ext            TEXT,
        semantic_type       TEXT,
        token_count         INTEGER,
        quality_score       DOUBLE PRECISION,
        risk_score          DOUBLE PRECISION,
        embedding_model     TEXT,
        valid_from          DATE,
        valid_to            DATE,
        edition_label       TEXT,
        ingestion_status    TEXT NOT NULL,
        ingestion_reason    TEXT,
        ingestion_timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        is_active           BOOLEAN NOT NULL DEFAULT FALSE
    );

    ALTER TABLE document_versions
        ADD CONSTRAINT chk_ingestion_status
        CHECK (ingestion_status IN ('pending','ingesting','ready','failed'));

    CREATE UNIQUE INDEX ux_document_versions_doc_ver
        ON document_versions(document_id, version_number);

    CREATE UNIQUE INDEX ux_versions_doc_contenthash
        ON document_versions(document_id, content_hash);

    CREATE UNIQUE INDEX ux_one_active_version
        ON document_versions(document_id)
        WHERE is_active = TRUE;

    CREATE INDEX ix_versions_doc_active ON document_versions(document_id, is_active);

    CREATE TRIGGER trg_versions_updated
        BEFORE UPDATE ON document_versions
        FOR EACH ROW EXECUTE FUNCTION set_updated_at();
    """
    conn.execute(sql)
    conn.commit()


def migration_003_create_document_versions_table_down(conn: psycopg.Connection) -> None:
    """Drop document_versions table."""
    sql = "DROP TABLE IF EXISTS document_versions CASCADE;"
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 4: Chunks table
# ============================================================

def migration_004_create_chunks_table_up(conn: psycopg.Connection) -> None:
    """Create chunks table."""
    sql = """
    CREATE TABLE IF NOT EXISTS chunks (
        id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        version_id      UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
        chunk_index     INTEGER NOT NULL,
        clause_path     TEXT,
        standard_ref    TEXT,
        section         TEXT,
        page            INTEGER,
        source          TEXT,
        char_len        INTEGER,
        token_count     INTEGER,
        qdrant_point_id TEXT NOT NULL,
        embedding_model TEXT,
        is_deleted      BOOLEAN NOT NULL DEFAULT FALSE,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    CREATE UNIQUE INDEX ux_chunks_version_index ON chunks(version_id, chunk_index);
    CREATE INDEX ix_chunks_version      ON chunks(version_id);
    CREATE INDEX ix_chunks_clause       ON chunks(version_id, clause_path);
    CREATE INDEX ix_chunks_standard_ref ON chunks(standard_ref);
    CREATE INDEX ix_chunks_point_id     ON chunks(qdrant_point_id);
    """
    conn.execute(sql)
    conn.commit()


def migration_004_create_chunks_table_down(conn: psycopg.Connection) -> None:
    """Drop chunks table."""
    sql = "DROP TABLE IF EXISTS chunks CASCADE;"
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 5: Tags tables
# ============================================================

def migration_005_create_tags_tables_up(conn: psycopg.Connection) -> None:
    """Create document_tags and version_tags tables."""
    sql = """
    CREATE TABLE IF NOT EXISTS document_tags (
        id              BIGSERIAL PRIMARY KEY,
        document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        tag             TEXT NOT NULL
    );
    CREATE UNIQUE INDEX ux_document_tags_doc_tag ON document_tags(document_id, tag);
    CREATE INDEX ix_document_tags_tag ON document_tags(tag);

    CREATE TABLE IF NOT EXISTS version_tags (
        id              BIGSERIAL PRIMARY KEY,
        version_id      UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
        tag             TEXT NOT NULL
    );
    CREATE UNIQUE INDEX ux_version_tags_ver_tag ON version_tags(version_id, tag);
    CREATE INDEX ix_version_tags_tag ON version_tags(tag);
    """
    conn.execute(sql)
    conn.commit()


def migration_005_create_tags_tables_down(conn: psycopg.Connection) -> None:
    """Drop tags tables."""
    sql = """
    DROP TABLE IF EXISTS version_tags CASCADE;
    DROP TABLE IF EXISTS document_tags CASCADE;
    """
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 6: ACL table
# ============================================================

def migration_006_create_acl_table_up(conn: psycopg.Connection) -> None:
    """Create document_acl table."""
    sql = """
    CREATE TABLE IF NOT EXISTS document_acl (
        id              BIGSERIAL PRIMARY KEY,
        document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        principal_type  TEXT NOT NULL,
        principal_id    TEXT NOT NULL,
        permission      TEXT NOT NULL DEFAULT 'read',
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    ALTER TABLE document_acl ADD CONSTRAINT chk_doc_acl_principal
        CHECK (principal_type IN ('role','user','group'));

    CREATE UNIQUE INDEX ux_doc_acl_unique
        ON document_acl(document_id, principal_type, principal_id, permission);
    CREATE INDEX ix_doc_acl_document  ON document_acl(document_id);
    CREATE INDEX ix_doc_acl_principal ON document_acl(principal_type, principal_id);
    """
    conn.execute(sql)
    conn.commit()


def migration_006_create_acl_table_down(conn: psycopg.Connection) -> None:
    """Drop document_acl table."""
    sql = "DROP TABLE IF EXISTS document_acl CASCADE;"
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 7: Views for ACL
# ============================================================

def migration_007_create_acl_views_up(conn: psycopg.Connection) -> None:
    """Create views for ACL and authorization."""
    sql = """
    CREATE OR REPLACE VIEW document_restriction AS
    SELECT
        d.id AS document_id,
        EXISTS (SELECT 1 FROM document_acl a WHERE a.document_id = d.id) AS is_restricted
    FROM documents d;

    CREATE OR REPLACE VIEW document_principals AS
    SELECT
        document_id,
        array_agg(principal_id) AS principals
    FROM document_acl
    WHERE permission = 'read'
    GROUP BY document_id;
    """
    conn.execute(sql)
    conn.commit()


def migration_007_create_acl_views_down(conn: psycopg.Connection) -> None:
    """Drop ACL views."""
    sql = """
    DROP VIEW IF EXISTS document_principals CASCADE;
    DROP VIEW IF EXISTS document_restriction CASCADE;
    """
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 8: Authorization function
# ============================================================

def migration_008_create_authorization_function_up(conn: psycopg.Connection) -> None:
    """Create authorized_chunk_ids function for access control."""
    sql = """
    CREATE OR REPLACE FUNCTION authorized_chunk_ids(
        p_user_id TEXT,
        p_roles   TEXT[]
    )
    RETURNS TABLE(chunk_id UUID) AS $$
        SELECT c.id
        FROM chunks c
        JOIN document_versions v ON v.id = c.version_id
        JOIN documents d         ON d.id = v.document_id
        WHERE c.is_deleted = FALSE
          AND (
                NOT EXISTS (SELECT 1 FROM document_acl a WHERE a.document_id = d.id)
                OR EXISTS (
                    SELECT 1 FROM document_acl a
                    WHERE a.document_id = d.id
                      AND a.permission = 'read'
                      AND (
                            (a.principal_type = 'user'  AND a.principal_id = p_user_id)
                         OR (a.principal_type IN ('role','group') AND a.principal_id = ANY(p_roles))
                      )
                )
          );
    $$ LANGUAGE sql STABLE;
    """
    conn.execute(sql)
    conn.commit()


def migration_008_create_authorization_function_down(conn: psycopg.Connection) -> None:
    """Drop authorized_chunk_ids function."""
    sql = "DROP FUNCTION IF EXISTS authorized_chunk_ids(TEXT, TEXT[]) CASCADE;"
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 9: Path history table
# ============================================================

def migration_009_create_path_history_table_up(conn: psycopg.Connection) -> None:
    """Create document_path_history table."""
    sql = """
    CREATE TABLE IF NOT EXISTS document_path_history (
        id              BIGSERIAL PRIMARY KEY,
        document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        version_id      UUID REFERENCES document_versions(id) ON DELETE SET NULL,
        old_path        TEXT,
        new_path        TEXT NOT NULL,
        changed_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX ix_document_path_history_document_id ON document_path_history(document_id);
    """
    conn.execute(sql)
    conn.commit()


def migration_009_create_path_history_table_down(conn: psycopg.Connection) -> None:
    """Drop document_path_history table."""
    sql = "DROP TABLE IF EXISTS document_path_history CASCADE;"
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 10: Ingestion logs table
# ============================================================

def migration_010_create_ingestion_logs_table_up(conn: psycopg.Connection) -> None:
    """Create ingestion_logs table."""
    sql = """
    CREATE TABLE IF NOT EXISTS ingestion_logs (
        id              BIGSERIAL PRIMARY KEY,
        version_id      UUID NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
        stage           TEXT NOT NULL,
        status          TEXT NOT NULL,
        error_message   TEXT,
        created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );

    ALTER TABLE ingestion_logs ADD CONSTRAINT chk_ingestion_log_stage
        CHECK (stage IN ('parsing','ocr','chunking','embedding','qdrant_write'));
    ALTER TABLE ingestion_logs ADD CONSTRAINT chk_ingestion_log_status
        CHECK (status IN ('ok','failed'));

    CREATE INDEX ix_ingestion_logs_version_id ON ingestion_logs(version_id);
    """
    conn.execute(sql)
    conn.commit()


def migration_010_create_ingestion_logs_table_down(conn: psycopg.Connection) -> None:
    """Drop ingestion_logs table."""
    sql = "DROP TABLE IF EXISTS ingestion_logs CASCADE;"
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Step 11: Document subtype columns
# ============================================================

def migration_011_add_document_subtype_columns_up(conn: psycopg.Connection) -> None:
    """Add document_subtype columns for hybrid subtype classification output."""
    sql = """
    ALTER TABLE documents
        ADD COLUMN IF NOT EXISTS document_subtype TEXT;

    ALTER TABLE document_versions
        ADD COLUMN IF NOT EXISTS document_subtype TEXT;

    CREATE INDEX IF NOT EXISTS ix_documents_document_subtype
        ON documents(document_subtype);
    CREATE INDEX IF NOT EXISTS ix_document_versions_document_subtype
        ON document_versions(document_subtype);
    """
    conn.execute(sql)
    conn.commit()


def migration_011_add_document_subtype_columns_down(conn: psycopg.Connection) -> None:
    """Drop subtype columns."""
    sql = """
    DROP INDEX IF EXISTS ix_document_versions_document_subtype;
    DROP INDEX IF EXISTS ix_documents_document_subtype;

    ALTER TABLE document_versions
        DROP COLUMN IF EXISTS document_subtype;
    ALTER TABLE documents
        DROP COLUMN IF EXISTS document_subtype;
    """
    conn.execute(sql)
    conn.commit()


# ============================================================
# Migration Registry
# ============================================================

POSTGRES_MIGRATIONS: list[dict[str, Any]] = [
    {
        "version": 1,
        "description": "Create trigger function set_updated_at",
        "up": migration_001_create_trigger_function_up,
        "down": migration_001_create_trigger_function_down,
    },
    {
        "version": 2,
        "description": "Create documents table",
        "up": migration_002_create_documents_table_up,
        "down": migration_002_create_documents_table_down,
    },
    {
        "version": 3,
        "description": "Create document_versions table",
        "up": migration_003_create_document_versions_table_up,
        "down": migration_003_create_document_versions_table_down,
    },
    {
        "version": 4,
        "description": "Create chunks table",
        "up": migration_004_create_chunks_table_up,
        "down": migration_004_create_chunks_table_down,
    },
    {
        "version": 5,
        "description": "Create document_tags and version_tags tables",
        "up": migration_005_create_tags_tables_up,
        "down": migration_005_create_tags_tables_down,
    },
    {
        "version": 6,
        "description": "Create document_acl table",
        "up": migration_006_create_acl_table_up,
        "down": migration_006_create_acl_table_down,
    },
    {
        "version": 7,
        "description": "Create ACL views (document_restriction, document_principals)",
        "up": migration_007_create_acl_views_up,
        "down": migration_007_create_acl_views_down,
    },
    {
        "version": 8,
        "description": "Create authorization function (authorized_chunk_ids)",
        "up": migration_008_create_authorization_function_up,
        "down": migration_008_create_authorization_function_down,
    },
    {
        "version": 9,
        "description": "Create document_path_history table",
        "up": migration_009_create_path_history_table_up,
        "down": migration_009_create_path_history_table_down,
    },
    {
        "version": 10,
        "description": "Create ingestion_logs table",
        "up": migration_010_create_ingestion_logs_table_up,
        "down": migration_010_create_ingestion_logs_table_down,
    },
    {
        "version": 11,
        "description": "Add document_subtype columns to documents and document_versions",
        "up": migration_011_add_document_subtype_columns_up,
        "down": migration_011_add_document_subtype_columns_down,
    },
]


def run_postgres_migrations(dsn: str, target_version: int | None = None) -> int:
    """
    Run PostgreSQL migrations up to target_version (or all if None).

    Returns the current migration version after execution.
    """
    current = 0
    try:
        with create_postgres_connection(dsn) as conn:
            for migration in POSTGRES_MIGRATIONS:
                if target_version and migration["version"] > target_version:
                    break
                migration["up"](conn)
                current = migration["version"]
        return current
    except Exception as e:
        raise RuntimeError(f"PostgreSQL migration failed at version {current}: {e}") from e


def rollback_postgres_migrations(dsn: str, target_version: int) -> int:
    """
    Rollback PostgreSQL migrations to target_version.

    Returns the current migration version after rollback.
    """
    current = len(POSTGRES_MIGRATIONS)
    try:
        with create_postgres_connection(dsn) as conn:
            for migration in reversed(POSTGRES_MIGRATIONS):
                if migration["version"] <= target_version:
                    break
                migration["down"](conn)
                current = migration["version"] - 1
        return current
    except Exception as e:
        raise RuntimeError(f"PostgreSQL rollback failed: {e}") from e
