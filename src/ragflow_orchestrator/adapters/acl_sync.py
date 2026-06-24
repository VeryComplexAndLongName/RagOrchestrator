# app/rag/acl_sync.py
from __future__ import annotations

from dataclasses import dataclass

import psycopg  # psycopg 3, как в зависимостях модуля


@dataclass(frozen=True)
class AclState:
    document_id: str
    is_restricted: bool
    principals: list[str]


class AclSync:
    """
    Синхронизатор ACL между PostgreSQL (источник истины) и Qdrant (проекция).

    Одним вызовом: обновляет права документа в Postgres в транзакции,
    перечитывает эффективное состояние из вьюх (document_restriction /
    document_principals) и проецирует его в payload всех чанков документа
    через QdrantProvider.update_acl_by_document — без реэмбеддинга.

    Политика: "нет записей в document_acl = документ доступен всем".
    """

    def __init__(self, dsn: str, provider) -> None:
        # provider — экземпляр QdrantProvider (или совместимый, c update_acl_by_document)
        self._dsn = dsn
        self._provider = provider

    # ----------------------------------------------------------
    # Публичные операции
    # ----------------------------------------------------------
    def grant(
        self,
        document_id: str,
        principal_id: str,
        principal_type: str = "role",
        permission: str = "read",
    ) -> AclState:
        """Выдать доступ принципалу и сразу спроецировать в Qdrant."""
        with psycopg.connect(self._dsn) as conn:
            with conn.transaction():
                conn.execute(
                    """
                    INSERT INTO document_acl
                        (document_id, principal_type, principal_id, permission)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (document_id, principal_type, principal_id, permission)
                    DO NOTHING
                    """,
                    (document_id, principal_type, principal_id, permission),
                )
                state = self._read_state(conn, document_id)
            # транзакция Postgres зафиксирована -> проецируем в Qdrant
            self._project(state)
        return state

    def revoke(
        self,
        document_id: str,
        principal_id: str,
        principal_type: str = "role",
        permission: str = "read",
    ) -> AclState:
        """Отозвать доступ у принципала и спроецировать в Qdrant."""
        with psycopg.connect(self._dsn) as conn:
            with conn.transaction():
                conn.execute(
                    """
                    DELETE FROM document_acl
                    WHERE document_id = %s
                      AND principal_type = %s
                      AND principal_id = %s
                      AND permission = %s
                    """,
                    (document_id, principal_type, principal_id, permission),
                )
                state = self._read_state(conn, document_id)
            self._project(state)
        return state

    def set_principals(
        self,
        document_id: str,
        principals: list[str],
        principal_type: str = "role",
        permission: str = "read",
    ) -> AclState:
        """
        Полностью заменить набор принципалов документа (для read-доступа
        указанного типа). Пустой список -> документ снова открыт для всех.
        """
        with psycopg.connect(self._dsn) as conn:
            with conn.transaction():
                conn.execute(
                    """
                    DELETE FROM document_acl
                    WHERE document_id = %s
                      AND principal_type = %s
                      AND permission = %s
                    """,
                    (document_id, principal_type, permission),
                )
                if principals:
                    conn.executemany(
                        """
                        INSERT INTO document_acl
                            (document_id, principal_type, principal_id, permission)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        [
                            (document_id, principal_type, p, permission)
                            for p in principals
                        ],
                    )
                state = self._read_state(conn, document_id)
            self._project(state)
        return state

    def resync(self, document_id: str) -> AclState:
        """
        Привести Qdrant в соответствие с текущим состоянием Postgres,
        ничего не меняя в правах. Полезно после реингеста документа
        или для устранения рассинхрона.
        """
        with psycopg.connect(self._dsn) as conn:
            state = self._read_state(conn, document_id)
        self._project(state)
        return state

    # ----------------------------------------------------------
    # Внутреннее
    # ----------------------------------------------------------
    @staticmethod
    def _read_state(conn: "psycopg.Connection", document_id: str) -> AclState:
        """Читает эффективное ACL-состояние документа из вьюх."""
        is_restricted = conn.execute(
            "SELECT is_restricted FROM document_restriction WHERE document_id = %s",
            (document_id,),
        ).fetchone()
        # документ может отсутствовать во вьюхе только если его нет в documents
        restricted = bool(is_restricted[0]) if is_restricted else False

        row = conn.execute(
            "SELECT principals FROM document_principals WHERE document_id = %s",
            (document_id,),
        ).fetchone()
        principals = list(row[0]) if row and row[0] else []

        return AclState(
            document_id=document_id,
            is_restricted=restricted,
            principals=principals,
        )

    def _project(self, state: AclState) -> None:
        """Проекция состояния в payload всех чанков документа (без реэмбеддинга)."""
        self._provider.update_acl_by_document(
            document_id=state.document_id,
            is_restricted=state.is_restricted,
            principals=state.principals,
        )