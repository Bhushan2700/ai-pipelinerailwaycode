import asyncio
import time
import random
from typing import Any, AsyncGenerator, Optional
import httpx
from core.config import FileMakerSettings
from core.exceptions import FileMakerAuthError, FileMakerFetchError, FileMakerStoreError
from core.logging import get_logger

logger = get_logger(__name__)


class FileMakerClient:
    def __init__(self, settings: FileMakerSettings) -> None:
        self._settings = settings
        self._token: Optional[str] = None
        self._token_acquired_at: float = 0.0
        self._client: Optional[httpx.AsyncClient] = None
        self._consecutive_failures: int = 0
        self._circuit_open_until: float = 0.0

    async def __aenter__(self) -> "FileMakerClient":
        self._client = httpx.AsyncClient(
            base_url=self._settings.host,
            timeout=httpx.Timeout(30.0),
            verify=True,
        )
        await self._acquire_token()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._release_token()
        if self._client:
            await self._client.aclose()

    async def _acquire_token(self) -> None:
        import base64
        credentials = base64.b64encode(
            f"{self._settings.username}:{self._settings.password}".encode()
        ).decode()
        try:
            resp = await self._request_with_retries(
                "POST",
                f"/fmi/data/v2/databases/{self._settings.database}/sessions",
                headers={"Authorization": f"Basic {credentials}", "Content-Type": "application/json"},
                json={},
                allow_auth_refresh=False,
            )
            resp.raise_for_status()
            self._token = resp.json()["response"]["token"]
            self._token_acquired_at = time.time()
            self._consecutive_failures = 0
            logger.info("filemaker_token_acquired")
        except httpx.HTTPStatusError as exc:
            raise FileMakerAuthError(f"Token acquisition failed: {exc.response.status_code}") from exc

    async def _release_token(self) -> None:
        if not self._token:
            return
        try:
            await self._client.delete(
                f"/fmi/data/v2/databases/{self._settings.database}/sessions/{self._token}",
                headers=self._auth_headers(),
            )
        except Exception:
            pass
        finally:
            self._token = None

    async def _ensure_valid_token(self) -> None:
        elapsed = time.time() - self._token_acquired_at
        if elapsed >= self._settings.token_refresh_interval:
            await self._acquire_token()

    def _check_circuit(self) -> None:
        if self._circuit_open_until and time.time() < self._circuit_open_until:
            wait_seconds = round(self._circuit_open_until - time.time(), 2)
            raise FileMakerFetchError(f"FileMaker circuit open; retry in ~{wait_seconds}s")
        if self._circuit_open_until and time.time() >= self._circuit_open_until:
            self._circuit_open_until = 0.0
            self._consecutive_failures = 0

    def _record_failure(self, reason: str) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self._settings.circuit_breaker_failures:
            self._circuit_open_until = time.time() + self._settings.circuit_breaker_reset_seconds
            logger.error(
                "filemaker_circuit_opened",
                reason=reason,
                consecutive_failures=self._consecutive_failures,
                open_for_seconds=self._settings.circuit_breaker_reset_seconds,
            )

    async def _request_with_retries(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
        allow_auth_refresh: bool = True,
    ) -> httpx.Response:
        self._check_circuit()
        last_error: Optional[Exception] = None
        for attempt in range(self._settings.max_retries + 1):
            try:
                resp = await self._client.request(
                    method,
                    url,
                    headers=headers,
                    params=params,
                    json=json,
                )

                if resp.status_code == 401 and allow_auth_refresh and self._token:
                    await self._acquire_token()
                    continue

                if resp.status_code in (429, 500, 502, 503, 504):
                    raise httpx.HTTPStatusError(
                        f"Retryable FileMaker status={resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )

                self._consecutive_failures = 0
                return resp
            except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
                last_error = exc
                self._record_failure(type(exc).__name__)
                if attempt >= self._settings.max_retries:
                    break
                backoff = min(
                    self._settings.retry_max_backoff_seconds,
                    self._settings.retry_backoff_base ** attempt,
                )
                jitter = random.uniform(0.0, 0.4)
                await asyncio.sleep(backoff + jitter)
        assert last_error is not None
        raise last_error

    def _auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._token}", "Content-Type": "application/json"}

    async def fetch_all_records(
        self,
        layout: str,
        offset: int = 1,
        limit: int = 100,
    ) -> AsyncGenerator[list[dict[str, Any]], None]:
        await self._ensure_valid_token()
        current_offset = offset
        while True:
            try:
                resp = await self._request_with_retries(
                    "GET",
                    f"/fmi/data/v2/databases/{self._settings.database}/layouts/{layout}/records",
                    headers=self._auth_headers(),
                    params={"_offset": current_offset, "_limit": limit},
                )
                resp.raise_for_status()
                data = resp.json()["response"]
                records: list[dict] = data.get("data", [])
                if not records:
                    break
                yield records
                if len(records) < limit:
                    break
                current_offset += limit
            except httpx.HTTPStatusError as exc:
                raise FileMakerFetchError(f"Fetch failed at offset {current_offset}: {exc}") from exc

    async def find_records(
        self,
        layout: str,
        query: list[dict[str, Any]],
        sort: Optional[list[dict[str, str]]] = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]:
        await self._ensure_valid_token()
        body: dict[str, Any] = {"query": query, "limit": str(limit)}
        if sort:
            body["sort"] = sort
        try:
            resp = await self._request_with_retries(
                "POST",
                f"/fmi/data/v2/databases/{self._settings.database}/layouts/{layout}/_find",
                headers=self._auth_headers(),
                json=body,
            )
            # FileMaker returns HTTP 200 with error code 401 in body when no records match
            # It may also return HTTP 400 depending on server version — both mean "no records"
            if resp.status_code == 400:
                body_data = resp.json()
                code = body_data.get("messages", [{}])[0].get("code", "")
                if code == "401":
                    return []
                raise FileMakerFetchError(f"Find request failed (FM code {code}): {body_data}")
            resp.raise_for_status()
            body_data = resp.json()
            # Check for FileMaker-level "no records" in a 200 response
            fm_code = body_data.get("messages", [{}])[0].get("code", "0")
            if fm_code == "401":
                return []
            return body_data["response"].get("data", [])
        except httpx.HTTPStatusError as exc:
            raise FileMakerFetchError(f"Find request failed: {exc}") from exc

    async def fetch_event_responses(self, event_id: str) -> dict[str, Any]:
        """
        Fetch and parse the N8N_SURVEY_EVENTS record for a given event_id.
        Always extracts Numeric_JSON and Text_JSON from the single FileMaker record.

        Returns:
            {
                "meta":               dict  — from asJSON field
                "numeric_questions":  dict  — from Numeric_JSON (structured + explanations)
                "text_questions":     list  — from Text_JSON (pure open-ended)
                "record_id":          str   — FileMaker record ID
                "survey_id":          str
                "industry":           str
                "survey_name":        str
                "survey_description": str
                "survey_recipients":  str
            }
        """
        import json as _json

        logger.info("fetching_event_responses", event_id=event_id)

        records = await self.find_records(
            layout="N8N_SURVEY_EVENTS",
            query=[{"ID_Event": f"=={event_id}"}],
            limit=1,
        )

        if not records:
            logger.warning("no_event_record_found", event_id=event_id)
            return {}

        fd        = records[0].get("fieldData", {})
        record_id = str(records[0].get("recordId", ""))

        meta             = _json.loads(fd.get("Meta", "{}"))
        numeric_raw      = _json.loads(fd.get("Numeric_JSON", "{}"))
        text_raw         = _json.loads(fd.get("Text_JSON", "{}"))

        numeric_questions = numeric_raw.get("questions", {})
        text_questions    = text_raw.get("text_responses", [])

        logger.info(
            "event_data_parsed",
            event_id=event_id,
            record_id=record_id,
            numeric_questions=len(numeric_questions),
            text_questions=len(text_questions),
        )

        return {
            "meta":               meta,
            "numeric_questions":  numeric_questions,
            "text_questions":     text_questions,
            "record_id":          record_id,
            "survey_id":          fd.get("ID_Survey", meta.get("id_survey", "")),
            "industry":           numeric_raw.get("industry", ""),
            "survey_name":        numeric_raw.get("survey_name", meta.get("surveyName", "")),
            "survey_description": numeric_raw.get("survey_description", ""),
            "survey_recipients":  numeric_raw.get("survey_recipients", "0"),
        }

    async def ping(self) -> dict[str, Any]:
        """Test connection by fetching database metadata."""
        await self._ensure_valid_token()
        try:
            resp = await self._request_with_retries(
                "GET",
                f"/fmi/data/v2/databases/{self._settings.database}/layouts",
                headers=self._auth_headers(),
            )
            resp.raise_for_status()
            layouts = [l["name"] for l in resp.json()["response"].get("layouts", [])]
            return {"connected": True, "layouts_accessible": len(layouts)}
        except httpx.HTTPStatusError as exc:
            raise FileMakerFetchError(f"Ping failed: {exc}") from exc

    async def create_record(self, layout: str, field_data: dict[str, Any]) -> str:
        await self._ensure_valid_token()
        try:
            resp = await self._request_with_retries(
                "POST",
                f"/fmi/data/v2/databases/{self._settings.database}/layouts/{layout}/records",
                headers=self._auth_headers(),
                json={"fieldData": field_data},
            )
            resp.raise_for_status()
            return str(resp.json()["response"]["recordId"])
        except httpx.HTTPStatusError as exc:
            raise FileMakerStoreError(f"Create record failed: {exc}") from exc

    async def update_record(self, layout: str, record_id: str, field_data: dict[str, Any]) -> None:
        await self._ensure_valid_token()
        try:
            resp = await self._request_with_retries(
                "PATCH",
                f"/fmi/data/v2/databases/{self._settings.database}/layouts/{layout}/records/{record_id}",
                headers=self._auth_headers(),
                json={"fieldData": field_data},
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise FileMakerStoreError(f"Update record failed: {exc}") from exc

    async def bulk_create_records(
        self,
        layout: str,
        records: list[dict[str, Any]],
        batch_size: int = 20,
    ) -> list[str]:
        record_ids: list[str] = []
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            for record in batch:
                rid = await self.create_record(layout, record)
                record_ids.append(rid)
        return record_ids
