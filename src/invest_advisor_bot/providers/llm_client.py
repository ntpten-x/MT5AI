from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence

import httpx
from loguru import logger

from invest_advisor_bot.observability import log_event
from invest_advisor_bot.runtime_diagnostics import diagnostics

DEFAULT_OPENROUTER_FREE_MODELS: tuple[str, ...] = (
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
    "mistralai/mistral-small-3.1-24b-instruct:free",
)
DEFAULT_GEMINI_MODELS: tuple[str, ...] = (
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
)
DEFAULT_GROQ_MODELS: tuple[str, ...] = (
    "llama-3.3-70b-versatile",
    "llama3-8b-8192",
)
DEFAULT_GITHUB_MODELS: tuple[str, ...] = (
    "openai/gpt-4.1-mini",
    "meta/Llama-3.3-70B-Instruct",
)
DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3
DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS = 180

ProviderKind = Literal["openai_compatible", "gemini_native"]


@dataclass(slots=True, frozen=True)
class LLMTextResponse:
    text: str
    model: str
    response_id: str | None
    raw_response: dict[str, Any]


@dataclass(slots=True, frozen=True)
class LLMProviderConfig:
    provider_name: str
    provider_kind: ProviderKind
    api_key: str
    base_url: str
    primary_model: str
    fallback_models: tuple[str, ...] = ()
    max_output_tokens: int = 600
    timeout: float = 30.0
    max_retries: int = 1
    endpoint_path: str = "/chat/completions"
    organization: str | None = None
    project: str | None = None
    http_referer: str | None = None
    app_title: str | None = None
    extra_headers: tuple[tuple[str, str], ...] = ()
    metadata_fields: tuple[str, ...] = field(default_factory=tuple)

    def available_models(self) -> tuple[str, ...]:
        models = [self.primary_model.strip(), *[model.strip() for model in self.fallback_models]]
        deduped: list[str] = []
        for model in models:
            if model and model not in deduped:
                deduped.append(model)
        return tuple(deduped)


class OpenAICompatibleLLMClient:
    """Multi-provider async client for OpenAI-compatible APIs and Gemini native REST."""

    def __init__(self, providers: Sequence[LLMProviderConfig]) -> None:
        self.providers = tuple(provider for provider in providers if provider.api_key.strip())
        self._provider_failures: dict[str, int] = {}
        self._provider_open_until: dict[str, float] = {}
        self._http_clients: dict[float, httpx.AsyncClient] = {}

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> LLMTextResponse | None:
        if not self.providers:
            logger.warning("No LLM providers are configured; skipping LLM request")
            log_event("llm_no_provider_configured", service=(metadata or {}).get("service"))
            return None

        for provider in self.providers:
            if self._is_circuit_open(provider.provider_name):
                continue
            provider_models = (model,) if model else provider.available_models()
            provider_succeeded = False
            for effective_model in provider_models:
                if not effective_model:
                    continue
                response = await self._call_provider(
                    provider=provider,
                    model=effective_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    metadata=metadata,
                )
                if response is not None:
                    provider_succeeded = True
                    self._reset_provider_failure(provider.provider_name)
                    log_event(
                        "llm_provider_success",
                        provider=provider.provider_name,
                        model=response.model,
                        service=(metadata or {}).get("service"),
                    )
                    diagnostics.record_provider_success(
                        provider=provider.provider_name,
                        model=response.model,
                        service=str((metadata or {}).get("service") or "unknown"),
                    )
                    return response
            if not provider_succeeded:
                self._record_provider_failure(provider.provider_name)
        log_event("llm_all_providers_failed", service=(metadata or {}).get("service"))
        return None

    async def _call_provider(
        self,
        *,
        provider: LLMProviderConfig,
        model: str,
        system_prompt: str,
        user_prompt: str,
        metadata: Mapping[str, str] | None,
    ) -> LLMTextResponse | None:
        if provider.provider_kind == "gemini_native":
            return await self._call_gemini_provider(
                provider=provider,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        return await self._call_openai_compatible_provider(
            provider=provider,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=metadata,
        )

    async def _call_openai_compatible_provider(
        self,
        *,
        provider: LLMProviderConfig,
        model: str,
        system_prompt: str,
        user_prompt: str,
        metadata: Mapping[str, str] | None,
    ) -> LLMTextResponse | None:
        payload: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": provider.max_output_tokens,
        }
        if metadata and provider.metadata_fields:
            payload["metadata"] = {
                key: value
                for key, value in metadata.items()
                if key in provider.metadata_fields and value.strip()
            }

        headers = {
            "Authorization": f"Bearer {provider.api_key}",
            "Content-Type": "application/json",
        }
        if provider.organization:
            headers["OpenAI-Organization"] = provider.organization
        if provider.project:
            headers["OpenAI-Project"] = provider.project
        if provider.http_referer:
            headers["HTTP-Referer"] = provider.http_referer
        if provider.app_title:
            headers["X-Title"] = provider.app_title
        for header_name, header_value in provider.extra_headers:
            if header_value.strip():
                headers[header_name] = header_value

        endpoint = f"{provider.base_url.rstrip('/')}{provider.endpoint_path}"
        response_json = await self._post_json(
            provider=provider,
            model=model,
            endpoint=endpoint,
            headers=headers,
            payload=payload,
        )
        if response_json is None:
            return None

        text = self._extract_openai_compatible_text(response_json)
        if not text:
            logger.warning("{} model {} did not include text output", provider.provider_name, model)
            return None
        return LLMTextResponse(
            text=text,
            model=str(response_json.get("model") or model),
            response_id=self._as_optional_str(response_json.get("id")),
            raw_response=dict(response_json),
        )

    async def _call_gemini_provider(
        self,
        *,
        provider: LLMProviderConfig,
        model: str,
        system_prompt: str,
        user_prompt: str,
    ) -> LLMTextResponse | None:
        endpoint = f"{provider.base_url.rstrip('/')}/models/{model}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": provider.max_output_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}
        params = {"key": provider.api_key}
        response_json = await self._post_json(
            provider=provider,
            model=model,
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            params=params,
        )
        if response_json is None:
            return None

        text = self._extract_gemini_text(response_json)
        if not text:
            logger.warning("{} model {} did not include text output", provider.provider_name, model)
            return None
        return LLMTextResponse(
            text=text,
            model=model,
            response_id=self._as_optional_str(response_json.get("responseId")),
            raw_response=dict(response_json),
        )

    async def _post_json(
        self,
        *,
        provider: LLMProviderConfig,
        model: str,
        endpoint: str,
        headers: Mapping[str, str],
        payload: Mapping[str, Any],
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        response: httpx.Response | None = None
        for attempt in range(provider.max_retries + 1):
            try:
                client = self._get_http_client(provider.timeout)
                response = await client.post(endpoint, headers=dict(headers), json=dict(payload), params=params)
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if attempt < provider.max_retries and status_code in {408, 409, 429, 500, 502, 503, 504}:
                    await asyncio.sleep(0.75 * (attempt + 1))
                    continue
                logger.warning(
                    "{} model {} returned HTTP {}: {}",
                    provider.provider_name,
                    model,
                    status_code,
                    exc.response.text,
                )
                log_event(
                    "llm_provider_http_error",
                    level="warning",
                    provider=provider.provider_name,
                    model=model,
                    status_code=status_code,
                )
                diagnostics.record_provider_failure(
                    provider=provider.provider_name,
                    model=model,
                    service=None,
                    detail={"status_code": status_code},
                )
                return None
            except httpx.HTTPError as exc:
                if attempt < provider.max_retries:
                    await asyncio.sleep(0.75 * (attempt + 1))
                    continue
                logger.warning("{} model {} request failed: {}", provider.provider_name, model, exc)
                log_event(
                    "llm_provider_request_failed",
                    level="warning",
                    provider=provider.provider_name,
                    model=model,
                    error=str(exc),
                )
                diagnostics.record_provider_failure(
                    provider=provider.provider_name,
                    model=model,
                    service=None,
                    detail={"error": str(exc)},
                )
                return None

        if response is None:
            return None
        try:
            parsed = response.json()
        except ValueError as exc:
            logger.warning("{} model {} returned invalid JSON: {}", provider.provider_name, model, exc)
            log_event(
                "llm_provider_invalid_json",
                level="warning",
                provider=provider.provider_name,
                model=model,
            )
            diagnostics.record_provider_failure(
                provider=provider.provider_name,
                model=model,
                service=None,
                detail={"error": "invalid_json"},
            )
            return None
        return dict(parsed) if isinstance(parsed, Mapping) else None

    async def aclose(self) -> None:
        clients = list(self._http_clients.values())
        self._http_clients.clear()
        for client in clients:
            await client.aclose()

    @staticmethod
    def _extract_openai_compatible_text(response_json: Mapping[str, Any]) -> str:
        choices = response_json.get("choices")
        if isinstance(choices, list):
            fragments: list[str] = []
            for choice in choices:
                if not isinstance(choice, Mapping):
                    continue
                message = choice.get("message")
                if not isinstance(message, Mapping):
                    continue
                content = message.get("content")
                if isinstance(content, str) and content.strip():
                    fragments.append(content.strip())
                    continue
                if isinstance(content, list):
                    for item in content:
                        if not isinstance(item, Mapping):
                            continue
                        text = item.get("text")
                        if isinstance(text, str) and text.strip():
                            fragments.append(text.strip())
            if fragments:
                return "\n".join(fragments).strip()

        output_text = response_json.get("output_text")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()
        return ""

    @staticmethod
    def _extract_gemini_text(response_json: Mapping[str, Any]) -> str:
        candidates = response_json.get("candidates")
        if not isinstance(candidates, list):
            return ""
        fragments: list[str] = []
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                continue
            content = candidate.get("content")
            if not isinstance(content, Mapping):
                continue
            parts = content.get("parts")
            if not isinstance(parts, list):
                continue
            for part in parts:
                if not isinstance(part, Mapping):
                    continue
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    fragments.append(text.strip())
        return "\n".join(fragments).strip()

    @staticmethod
    def _as_optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _is_circuit_open(self, provider_name: str) -> bool:
        open_until = self._provider_open_until.get(provider_name, 0.0)
        if open_until <= time.monotonic():
            if provider_name in self._provider_open_until:
                self._provider_open_until.pop(provider_name, None)
                diagnostics.record_provider_circuit(
                    provider=provider_name,
                    is_open=False,
                    failure_count=self._provider_failures.get(provider_name, 0),
                    open_until=None,
                )
            return False
        diagnostics.record_provider_circuit(
            provider=provider_name,
            is_open=True,
            failure_count=self._provider_failures.get(provider_name, 0),
            open_until=datetime.now(timezone.utc) + timedelta(seconds=max(0.0, open_until - time.monotonic())),
        )
        log_event("llm_provider_circuit_open", level="warning", provider=provider_name)
        return True

    def _record_provider_failure(self, provider_name: str) -> None:
        failure_count = self._provider_failures.get(provider_name, 0) + 1
        self._provider_failures[provider_name] = failure_count
        open_until_dt = None
        if failure_count >= DEFAULT_CIRCUIT_BREAKER_FAILURE_THRESHOLD:
            monotonic_until = time.monotonic() + DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS
            self._provider_open_until[provider_name] = monotonic_until
            open_until_dt = datetime.now(timezone.utc) + timedelta(seconds=DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS)
            log_event(
                "llm_provider_circuit_trip",
                level="warning",
                provider=provider_name,
                failure_count=failure_count,
                cooldown_seconds=DEFAULT_CIRCUIT_BREAKER_COOLDOWN_SECONDS,
            )
        diagnostics.record_provider_circuit(
            provider=provider_name,
            is_open=provider_name in self._provider_open_until,
            failure_count=failure_count,
            open_until=open_until_dt,
        )

    def _reset_provider_failure(self, provider_name: str) -> None:
        self._provider_failures[provider_name] = 0
        self._provider_open_until.pop(provider_name, None)
        diagnostics.record_provider_circuit(
            provider=provider_name,
            is_open=False,
            failure_count=0,
            open_until=None,
        )

    def _get_http_client(self, timeout: float) -> httpx.AsyncClient:
        key = float(timeout)
        client = self._http_clients.get(key)
        if client is None:
            client = httpx.AsyncClient(timeout=timeout)
            self._http_clients[key] = client
        return client


class OpenAILLMClient(OpenAICompatibleLLMClient):
    """Backwards-compatible single-provider OpenAI-compatible wrapper."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        max_output_tokens: int = 1_000,
        organization: str | None = None,
        project: str | None = None,
        max_retries: int = 1,
    ) -> None:
        provider = LLMProviderConfig(
            provider_name="openai",
            provider_kind="openai_compatible",
            api_key=api_key.strip(),
            base_url=base_url.rstrip("/"),
            primary_model=model,
            max_output_tokens=max_output_tokens,
            timeout=timeout,
            organization=organization,
            project=project,
            max_retries=max(0, int(max_retries)),
        )
        super().__init__([provider])


def build_default_llm_client(
    *,
    llm_api_key: str,
    llm_model: str,
    llm_base_url: str,
    llm_timeout_seconds: float,
    llm_max_output_tokens: int,
    llm_organization: str | None,
    llm_project: str | None,
    llm_provider: str = "auto",
    llm_provider_order: Sequence[str] | str = (),
    llm_model_fallbacks: Sequence[str] = (),
    openrouter_api_key: str = "",
    openrouter_models: Sequence[str] = DEFAULT_OPENROUTER_FREE_MODELS,
    openrouter_base_url: str = "https://openrouter.ai/api/v1",
    openrouter_http_referer: str | None = None,
    openrouter_app_title: str | None = None,
    gemini_api_key: str = "",
    gemini_models: Sequence[str] = DEFAULT_GEMINI_MODELS,
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta",
    groq_api_key: str = "",
    groq_models: Sequence[str] = DEFAULT_GROQ_MODELS,
    groq_base_url: str = "https://api.groq.com/openai/v1",
    github_models_api_key: str = "",
    github_models: Sequence[str] = DEFAULT_GITHUB_MODELS,
    github_models_base_url: str = "https://models.github.ai/inference",
    github_models_api_version: str = "2026-03-10",
) -> OpenAICompatibleLLMClient:
    provider_order = _normalize_provider_order(llm_provider=llm_provider, llm_provider_order=llm_provider_order)
    provider_map: dict[str, LLMProviderConfig] = {
        "gemini": LLMProviderConfig(
            provider_name="gemini",
            provider_kind="gemini_native",
            api_key=gemini_api_key.strip(),
            base_url=gemini_base_url.rstrip("/"),
            primary_model=(tuple(gemini_models) or DEFAULT_GEMINI_MODELS)[0],
            fallback_models=tuple(gemini_models[1:]) if gemini_models else DEFAULT_GEMINI_MODELS[1:],
            timeout=llm_timeout_seconds,
            max_output_tokens=llm_max_output_tokens,
            max_retries=1,
        ),
        "openrouter": LLMProviderConfig(
            provider_name="openrouter",
            provider_kind="openai_compatible",
            api_key=openrouter_api_key.strip(),
            base_url=openrouter_base_url.rstrip("/"),
            primary_model=(tuple(openrouter_models) or DEFAULT_OPENROUTER_FREE_MODELS)[0],
            fallback_models=tuple(openrouter_models[1:]) if openrouter_models else DEFAULT_OPENROUTER_FREE_MODELS[1:],
            timeout=llm_timeout_seconds,
            max_output_tokens=llm_max_output_tokens,
            max_retries=1,
            http_referer=openrouter_http_referer,
            app_title=openrouter_app_title or "Invest Advisor Bot",
            metadata_fields=("service", "language", "profile", "scope"),
        ),
        "groq": LLMProviderConfig(
            provider_name="groq",
            provider_kind="openai_compatible",
            api_key=groq_api_key.strip(),
            base_url=groq_base_url.rstrip("/"),
            primary_model=(tuple(groq_models) or DEFAULT_GROQ_MODELS)[0],
            fallback_models=tuple(groq_models[1:]) if groq_models else DEFAULT_GROQ_MODELS[1:],
            timeout=llm_timeout_seconds,
            max_output_tokens=llm_max_output_tokens,
            max_retries=1,
        ),
        "github_models": LLMProviderConfig(
            provider_name="github_models",
            provider_kind="openai_compatible",
            api_key=github_models_api_key.strip(),
            base_url=github_models_base_url.rstrip("/"),
            primary_model=(tuple(github_models) or DEFAULT_GITHUB_MODELS)[0],
            fallback_models=tuple(github_models[1:]) if github_models else DEFAULT_GITHUB_MODELS[1:],
            timeout=llm_timeout_seconds,
            max_output_tokens=llm_max_output_tokens,
            max_retries=1,
            extra_headers=(
                ("Accept", "application/vnd.github+json"),
                ("X-GitHub-Api-Version", github_models_api_version),
            ),
        ),
        "openai": LLMProviderConfig(
            provider_name="openai",
            provider_kind="openai_compatible",
            api_key=llm_api_key.strip(),
            base_url=llm_base_url.rstrip("/"),
            primary_model=llm_model,
            fallback_models=tuple(model for model in llm_model_fallbacks if model.strip()),
            timeout=llm_timeout_seconds,
            max_output_tokens=llm_max_output_tokens,
            max_retries=1,
            organization=llm_organization,
            project=llm_project,
        ),
    }

    providers = [provider_map[name] for name in provider_order if name in provider_map]
    return OpenAICompatibleLLMClient(providers)


def _normalize_provider_order(
    *,
    llm_provider: str,
    llm_provider_order: Sequence[str] | str,
) -> tuple[str, ...]:
    if isinstance(llm_provider_order, str):
        explicit_order = [item.strip().casefold() for item in llm_provider_order.split(",") if item.strip()]
    else:
        explicit_order = [str(item).strip().casefold() for item in llm_provider_order if str(item).strip()]
    if explicit_order:
        return tuple(dict.fromkeys(explicit_order))

    normalized_provider = llm_provider.strip().casefold() or "auto"
    if normalized_provider == "gemini":
        return ("gemini", "openrouter", "groq", "github_models", "openai")
    if normalized_provider == "groq":
        return ("groq", "gemini", "openrouter", "github_models", "openai")
    if normalized_provider == "github_models":
        return ("github_models", "gemini", "openrouter", "groq", "openai")
    if normalized_provider == "openrouter":
        return ("openrouter", "gemini", "groq", "github_models", "openai")
    if normalized_provider == "openai":
        return ("openai", "gemini", "openrouter", "groq", "github_models")
    return ("gemini", "openrouter", "groq", "github_models", "openai")
