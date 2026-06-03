from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Tuple
from urllib import request, parse

from ragflow_orchestrator.templates.base import BaseIngestionTemplate
from ragflow_orchestrator.templates.models import (
    BitrixConfig,
    TemplateRunReport,
    IngestionError,
)


class BitrixTemplate(BaseIngestionTemplate):
    """
    Ingestion-template for Bitrix24 via webhook API.

    Expects a BitrixConfig with:
      - domain: "<domain>.bitrix24.ru"
      - user_id: int
      - token: str

    The base URL is formed as:
      https://<domain>.bitrix24.ru/rest/<user_id>/<token>
    """

    # -------------------- public entrypoint --------------------

    def run(self, config: BitrixConfig) -> TemplateRunReport:
        report = TemplateRunReport()

        if config.include_contacts:
            self._run_entity(
                config=config,
                method="crm.contact.list",
                entity_name="contact",
                max_items=config.max_contacts,
                text_builder=self._contact_text,
                report=report,
            )

        if config.include_companies:
            self._run_entity(
                config=config,
                method="crm.company.list",
                entity_name="company",
                max_items=config.max_companies,
                text_builder=self._company_text,
                report=report,
            )

        if config.include_deals:
            self._run_entity(
                config=config,
                method="crm.deal.list",
                entity_name="deal",
                max_items=config.max_deals,
                text_builder=self._deal_text,
                report=report,
            )

        if config.include_leads:
            self._run_entity(
                config=config,
                method="crm.lead.list",
                entity_name="lead",
                max_items=config.max_leads,
                text_builder=self._lead_text,
                report=report,
            )

        if config.include_tasks:
            self._run_entity(
                config=config,
                method="tasks.task.list",
                entity_name="task",
                max_items=config.max_tasks,
                text_builder=self._task_text,
                report=report,
                items_key="result.tasks",
            )

        if config.include_activities:
            self._run_entity(
                config=config,
                method="crm.activity.list",
                entity_name="activity",
                max_items=config.max_activities,
                text_builder=self._activity_text,
                report=report,
            )

        if config.include_im_dialogs and config.dialog_ids:
            for dialog_id in config.dialog_ids:
                try:
                    messages = self._get_dialog_messages(
                        config=config,
                        dialog_id=dialog_id,
                        limit=config.max_dialog_messages,
                    )
                    if not messages:
                        report.skipped.append(
                            IngestionError(
                                source=f"im:{dialog_id}",
                                reason="no messages",
                            )
                        )
                        continue

                    text, metadata = self._im_dialog_text(dialog_id, messages)
                    if not text.strip():
                        report.skipped.append(
                            IngestionError(
                                source=f"im:{dialog_id}",
                                reason="empty dialog text",
                            )
                        )
                        continue

                    language = self._language_tag(
                        text=text,
                        mode=config.language_mode,
                    )
                    metadata["language"] = language

                    source_url = self._bitrix_source_url(config, "im", dialog_id)

                    summary = self.orchestrator.ingest(
                        source_id=f"bitrix:im:{dialog_id}",
                        raw_text=text,
                        metadata=self._metadata_for_url_source(
                            "bitrix",
                            metadata,
                            source_url,
                        ),
                    )
                    if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                        report.skipped.append(
                            IngestionError(
                                source=f"im:{dialog_id}",
                                reason="all chunks are duplicates",
                            )
                        )
                        continue
                    report.ingested.append(summary)
                except Exception as exc:  # pragma: no cover
                    report.failed.append(
                        IngestionError(
                            source=f"im:{dialog_id}",
                            reason=str(exc),
                        )
                    )

        return report

    # -------------------- common logic for list-method traversal --------------------

    def _run_entity(
        self,
        config: BitrixConfig,
        method: str,
        entity_name: str,
        max_items: int,
        text_builder,
        report: TemplateRunReport,
        items_key: str = "result",
    ) -> None:
        count = 0
        for item in self._paged_call(config, method=method, items_key=items_key):
            if count >= max_items:
                break
            count += 1
            try:
                entity_id = str(item.get("ID") or item.get("id") or "")
                if not entity_id:
                    report.skipped.append(
                        IngestionError(
                            source=f"{entity_name}:<no-id>",
                            reason="missing ID",
                        )
                    )
                    continue

                text, metadata = text_builder(item)
                if not text.strip():
                    report.skipped.append(
                        IngestionError(
                            source=f"{entity_name}:{entity_id}",
                            reason="empty text",
                        )
                    )
                    continue

                language = self._language_tag(
                    text=text,
                    mode=config.language_mode,
                )
                metadata["language"] = language

                source_url = self._bitrix_source_url(config, entity_name, entity_id)

                summary = self.orchestrator.ingest(
                    source_id=f"bitrix:{entity_name}:{entity_id}",
                    raw_text=text,
                    metadata=self._metadata_for_url_source(
                        "bitrix",
                        metadata,
                        source_url,
                    ),
                )
                if summary.total_chunks == 0 and summary.duplicate_chunks_skipped > 0:
                    report.skipped.append(
                        IngestionError(
                            source=f"{entity_name}:{entity_id}",
                            reason="all chunks are duplicates",
                        )
                    )
                    continue
                report.ingested.append(summary)
            except Exception as exc:  # pragma: no cover
                report.failed.append(
                    IngestionError(
                        source=f"{entity_name}:{item.get('ID') or item.get('id') or 'entity'}",
                        reason=str(exc),
                    )
                )

    def _paged_call(
        self,
        config: BitrixConfig,
        method: str,
        items_key: str = "result",
    ) -> Iterable[Dict[str, Any]]:
        start = 0
        while True:
            params = {"start": start}
            payload = self._request_json(config, method, params)
            items = payload.get(items_key) or []
            if isinstance(items, dict):
                items = list(items.values())
            if not items:
                break
            for item in items:
                yield item
            next_start = payload.get("next")
            if next_start is None:
                break
            start = int(next_start)

    # -------------------- HTTP --------------------

    @staticmethod
    def _request_json(
        config: BitrixConfig,
        method: str,
        params: Dict[str, Any] | None,
    ) -> Dict[str, Any]:
        url = f"{config.base_url}/{method}.json"
        data_bytes = json.dumps(params or {}).encode("utf-8")

        req = request.Request(url=url, data=data_bytes, method="POST")
        req.add_header("Accept", "application/json")
        req.add_header("Content-Type", "application/json")

        with request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    @staticmethod
    def _bitrix_source_url(
        config: BitrixConfig,
        kind: str,
        entity_id: str,
    ) -> str:
        base = f"https://{config.domain}".rstrip("/")
        if kind == "contact":
            return f"{base}/crm/contact/details/{entity_id}/"
        if kind == "company":
            return f"{base}/crm/company/details/{entity_id}/"
        if kind == "deal":
            return f"{base}/crm/deal/details/{entity_id}/"
        if kind == "lead":
            return f"{base}/crm/lead/details/{entity_id}/"
        if kind == "task":
            return f"{base}/company/personal/user/0/tasks/task/view/{entity_id}/"
        if kind == "activity":
            return f"{base}/crm/activity/?ID={entity_id}"
        if kind == "im":
            return f"{base}/online/?IM_DIALOG={parse.quote(entity_id)}"
        return base

    # -------------------- text builders --------------------

    @staticmethod
    def _join_nonempty(parts: List[str]) -> str:
        return "\n".join(p for p in parts if p)

    def _contact_text(self, contact: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        phones = ", ".join(
            p.get("VALUE") for p in contact.get("PHONE", []) if p.get("VALUE")
        )
        emails = ", ".join(
            e.get("VALUE") for e in contact.get("EMAIL", []) if e.get("VALUE")
        )

        lines = [
            "Тип: Контакт",
            f"ID: {contact.get('ID')}",
            f"Имя: {(contact.get('NAME') or '').strip()} {(contact.get('SECOND_NAME') or '').strip()} {(contact.get('LAST_NAME') or '').strip()}".strip(),
            f"Телефон(ы): {phones}" if phones else "",
            f"E-mail(ы): {emails}" if emails else "",
            f"Тип контакта: {contact.get('TYPE_ID') or ''}",
            f"Компания ID: {contact.get('COMPANY_ID') or ''}",
            "",
            "Комментарии:",
            contact.get("COMMENTS") or "",
        ]
        text = self._join_nonempty(lines)
        metadata: Dict[str, Any] = {
            "entity": "contact",
            "bitrix_id": str(contact.get("ID") or ""),
            "company_id": str(contact.get("COMPANY_ID") or ""),
            "assigned_by_id": str(contact.get("ASSIGNED_BY_ID") or ""),
        }
        return text, metadata

    def _company_text(self, company: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        phones = ", ".join(
            p.get("VALUE") for p in company.get("PHONE", []) if p.get("VALUE")
        )
        emails = ", ".join(
            e.get("VALUE") for e in company.get("EMAIL", []) if e.get("VALUE")
        )
        webs = ", ".join(
            w.get("VALUE") for w in company.get("WEB", []) if w.get("VALUE")
        )

        lines = [
            "Тип: Компания",
            f"ID: {company.get('ID')}",
            f"Название: {company.get('TITLE') or ''}",
            f"Тип компании: {company.get('COMPANY_TYPE') or ''}",
            f"Отрасль: {company.get('INDUSTRY') or ''}",
            f"Телефон(ы): {phones}" if phones else "",
            f"E-mail(ы): {emails}" if emails else "",
            f"Сайт(ы): {webs}" if webs else "",
            f"Адрес: {company.get('ADDRESS') or ''}",
            "",
            "Комментарии:",
            company.get("COMMENTS") or "",
        ]
        text = self._join_nonempty(lines)
        metadata: Dict[str, Any] = {
            "entity": "company",
            "bitrix_id": str(company.get("ID") or ""),
            "company_type": company.get("COMPANY_TYPE") or "",
            "industry": company.get("INDUSTRY") or "",
        }
        return text, metadata

    def _deal_text(self, deal: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        lines = [
            "Тип: Сделка",
            f"ID: {deal.get('ID')}",
            f"Название: {deal.get('TITLE') or ''}",
            f"Стадия: {deal.get('STAGE_ID') or ''}",
            f"Сумма: {deal.get('OPPORTUNITY')} {deal.get('CURRENCY_ID')}",
            f"Ответственный ID: {deal.get('ASSIGNED_BY_ID')}",
            f"Компания ID: {deal.get('COMPANY_ID')}",
            f"Контакт ID: {deal.get('CONTACT_ID')}",
            "",
            "Комментарии:",
            deal.get("COMMENTS") or "",
        ]
        text = self._join_nonempty(lines)
        metadata: Dict[str, Any] = {
            "entity": "deal",
            "bitrix_id": str(deal.get("ID") or ""),
            "stage_id": deal.get("STAGE_ID") or "",
            "company_id": str(deal.get("COMPANY_ID") or ""),
            "contact_id": str(deal.get("CONTACT_ID") or ""),
            "assigned_by_id": str(deal.get("ASSIGNED_BY_ID") or ""),
        }
        return text, metadata

    def _lead_text(self, lead: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        lines = [
            "Тип: Лид",
            f"ID: {lead.get('ID')}",
            f"Название: {lead.get('TITLE') or ''}",
            f"Статус: {lead.get('STATUS_ID') or ''}",
            f"Источник: {lead.get('SOURCE_ID') or ''}",
            f"Сумма: {lead.get('OPPORTUNITY')} {lead.get('CURRENCY_ID')}",
            f"Ответственный ID: {lead.get('ASSIGNED_BY_ID')}",
            f"Компания ID: {lead.get('COMPANY_ID')}",
            f"Контакт ID: {lead.get('CONTACT_ID')}",
            "",
            "Комментарии:",
            lead.get("COMMENTS") or "",
        ]
        text = self._join_nonempty(lines)
        metadata: Dict[str, Any] = {
            "entity": "lead",
            "bitrix_id": str(lead.get("ID") or ""),
            "status_id": lead.get("STATUS_ID") or "",
            "source_id": lead.get("SOURCE_ID") or "",
            "company_id": str(lead.get("COMPANY_ID") or ""),
            "contact_id": str(lead.get("CONTACT_ID") or ""),
            "assigned_by_id": str(lead.get("ASSIGNED_BY_ID") or ""),
        }
        return text, metadata

    def _task_text(self, task: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        core = task.get("task") or task
        lines = [
            "Тип: Задача",
            f"ID: {core.get('ID')}",
            f"Название: {core.get('TITLE') or ''}",
            "",
            "Описание:",
            core.get("DESCRIPTION") or "",
            "",
            f"Ответственный ID: {core.get('RESPONSIBLE_ID')}",
            f"Постановщик ID: {core.get('CREATED_BY')}",
            f"Группа ID: {core.get('GROUP_ID')}",
            f"Статус: {core.get('STATUS')}",
        ]
        text = self._join_nonempty(lines)
        metadata: Dict[str, Any] = {
            "entity": "task",
            "bitrix_id": str(core.get("ID") or ""),
            "responsible_id": str(core.get("RESPONSIBLE_ID") or ""),
            "created_by": str(core.get("CREATED_BY") or ""),
            "group_id": str(core.get("GROUP_ID") or ""),
            "status": str(core.get("STATUS") or ""),
        }
        return text, metadata

    def _activity_text(self, activity: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        lines = [
            "Тип: Активность CRM",
            f"ID: {activity.get('ID')}",
            f"Тип: {activity.get('TYPE_ID')}",
            f"Субъект: {activity.get('SUBJECT')}",
            f"Ответственный ID: {activity.get('RESPONSIBLE_ID')}",
            f"Связь с сущностью: {activity.get('OWNER_TYPE_ID')} / {activity.get('OWNER_ID')}",
            "",
            "Описание:",
            activity.get("DESCRIPTION") or "",
        ]
        text = self._join_nonempty(lines)
        metadata: Dict[str, Any] = {
            "entity": "activity",
            "bitrix_id": str(activity.get("ID") or ""),
            "type_id": activity.get("TYPE_ID") or "",
            "owner_type_id": activity.get("OWNER_TYPE_ID") or "",
            "owner_id": str(activity.get("OWNER_ID") or ""),
            "responsible_id": str(activity.get("RESPONSIBLE_ID") or ""),
        }
        return text, metadata

    def _im_dialog_text(
        self,
        dialog_id: str,
        messages: List[Dict[str, Any]],
    ) -> Tuple[str, Dict[str, Any]]:
        rows: List[str] = []
        for msg in messages:
            author = msg.get("AUTHOR_ID") or msg.get("author_id") or ""
            text = msg.get("MESSAGE") or msg.get("message") or ""
            ts = msg.get("DATE_CREATE") or msg.get("date") or ""
            rows.append(f"[{ts}] ({author}) {text}")

        lines = [
            "Тип: Чат",
            f"Dialog ID: {dialog_id}",
            "",
            "Сообщения:",
            "\n".join(rows),
        ]
        text = self._join_nonempty(lines)
        metadata: Dict[str, Any] = {
            "entity": "im_dialog",
            "dialog_id": dialog_id,
        }
        return text, metadata

    # -------------------- IM helper --------------------

    def _get_dialog_messages(
        self,
        config: BitrixConfig,
        dialog_id: str,
        limit: int,
    ) -> List[Dict[str, Any]]:
        payload = self._request_json(
            config,
            method="im.message.get",
            params={
                "DIALOG_ID": dialog_id,
                "LIMIT": limit,
                "ORDER": "ASC",
            },
        )
        return list(payload.get("result") or [])
