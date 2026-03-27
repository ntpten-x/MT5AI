from __future__ import annotations

import httpx
import pytest

import invest_advisor_bot.providers.llm_client as llm_mod


class _MockResponse:
    def __init__(self, *, status_code: int, payload: dict, text: str = "", headers: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = text or str(payload)
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "https://example.com")
            raise httpx.HTTPStatusError("error", request=request, response=self)

    def json(self) -> dict:
        return self._payload


class _MockAsyncClient:
    def __init__(self, *, queue: list[_MockResponse], calls: list[dict], **_: object) -> None:
        self._queue = queue
        self._calls = calls

    async def __aenter__(self) -> "_MockAsyncClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(
        self,
        url: str,
        *,
        headers: dict | None = None,
        json: dict | None = None,
        params: dict | None = None,
    ) -> _MockResponse:
        self._calls.append({"url": url, "headers": headers or {}, "json": json or {}, "params": params or {}})
        return self._queue.pop(0)


@pytest.mark.asyncio
async def test_llm_client_routes_to_gemini_first(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = [
        _MockResponse(
            status_code=200,
            payload={
                "responseId": "gem-1",
                "candidates": [
                    {"content": {"parts": [{"text": "สรุปจาก Gemini"}]}},
                ],
            },
        )
    ]
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_mod.httpx,
        "AsyncClient",
        lambda **kwargs: _MockAsyncClient(queue=queue, calls=calls, **kwargs),
    )

    client = llm_mod.build_default_llm_client(
        llm_api_key="",
        llm_model="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_timeout_seconds=10.0,
        llm_max_output_tokens=200,
        llm_organization=None,
        llm_project=None,
        llm_provider="auto",
        gemini_api_key="gem-key",
        gemini_models=("gemini-2.0-flash",),
    )

    result = await client.generate_text(system_prompt="sys", user_prompt="user")

    assert result is not None
    assert result.model == "gemini-2.0-flash"
    assert result.text == "สรุปจาก Gemini"
    assert calls[0]["url"].endswith("/models/gemini-2.0-flash:generateContent")
    assert calls[0]["params"]["key"] == "gem-key"


@pytest.mark.asyncio
async def test_llm_client_falls_back_across_provider_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = [
        _MockResponse(status_code=429, payload={"error": {"message": "rate limit"}}, text="rate limit"),
        _MockResponse(status_code=429, payload={"error": {"message": "rate limit"}}, text="rate limit"),
        _MockResponse(
            status_code=200,
            payload={
                "id": "groq-1",
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"content": "fallback success"}}],
            },
        ),
    ]
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_mod.httpx,
        "AsyncClient",
        lambda **kwargs: _MockAsyncClient(queue=queue, calls=calls, **kwargs),
    )

    client = llm_mod.build_default_llm_client(
        llm_api_key="",
        llm_model="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_timeout_seconds=10.0,
        llm_max_output_tokens=200,
        llm_organization=None,
        llm_project=None,
        llm_provider="auto",
        gemini_api_key="gem-key",
        gemini_models=("gemini-2.0-flash",),
        groq_api_key="groq-key",
        groq_models=("llama-3.3-70b-versatile",),
    )

    result = await client.generate_text(system_prompt="sys", user_prompt="user")

    assert result is not None
    assert result.model == "llama-3.3-70b-versatile"
    assert result.text == "fallback success"
    assert len(calls) == 3
    assert calls[0]["url"].endswith("/models/gemini-2.0-flash:generateContent")
    assert calls[1]["url"].endswith("/models/gemini-2.0-flash:generateContent")
    assert calls[2]["url"].endswith("/chat/completions")


@pytest.mark.asyncio
async def test_llm_client_sets_github_models_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = [
        _MockResponse(
            status_code=200,
            payload={
                "id": "gh-1",
                "model": "openai/gpt-4.1-mini",
                "choices": [{"message": {"content": "github ok"}}],
            },
        )
    ]
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_mod.httpx,
        "AsyncClient",
        lambda **kwargs: _MockAsyncClient(queue=queue, calls=calls, **kwargs),
    )

    client = llm_mod.build_default_llm_client(
        llm_api_key="",
        llm_model="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_timeout_seconds=10.0,
        llm_max_output_tokens=200,
        llm_organization=None,
        llm_project=None,
        llm_provider="github_models",
        github_models_api_key="gh-key",
        github_models=("openai/gpt-4.1-mini",),
    )

    result = await client.generate_text(system_prompt="sys", user_prompt="user")

    assert result is not None
    assert result.text == "github ok"
    assert calls[0]["headers"]["Authorization"] == "Bearer gh-key"
    assert calls[0]["headers"]["Accept"] == "application/vnd.github+json"
    assert calls[0]["headers"]["X-GitHub-Api-Version"] == "2026-03-10"


def test_llm_client_status_exposes_provider_chain() -> None:
    client = llm_mod.build_default_llm_client(
        llm_api_key="openai-key",
        llm_model="gpt-5-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_timeout_seconds=10.0,
        llm_max_output_tokens=200,
        llm_organization=None,
        llm_project=None,
        llm_provider="auto",
        gemini_api_key="gem-key",
        gemini_models=("gemini-2.0-flash",),
        groq_api_key="groq-key",
        groq_models=("llama-3.3-70b-versatile",),
    )

    status = client.status()

    assert status["available"] is True
    assert status["provider_order"] == ["gemini", "groq", "openai"]
    assert status["configured_providers"][0]["primary_model"] == "gemini-2.0-flash"
    assert status["configured_providers"][1]["provider"] == "groq"


def test_llm_client_status_includes_extended_free_provider_chain() -> None:
    client = llm_mod.build_default_llm_client(
        llm_api_key="openai-key",
        llm_model="gpt-5-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_timeout_seconds=10.0,
        llm_max_output_tokens=200,
        llm_organization=None,
        llm_project=None,
        llm_provider="auto",
        llm_provider_order=("gemini", "cerebras", "cloudflare", "huggingface", "openai"),
        gemini_api_key="gem-key",
        gemini_models=("gemini-2.0-flash",),
        cerebras_api_key="cerebras-key",
        cerebras_models=("gpt-oss-120b",),
        cloudflare_account_id="acc-123",
        cloudflare_api_token="cloudflare-token",
        cloudflare_models=("@cf/meta/llama-3.1-8b-instruct-fast",),
        huggingface_api_key="hf-key",
        huggingface_models=("google/gemma-2-2b-it",),
    )

    status = client.status()

    assert status["provider_order"] == ["gemini", "cerebras", "cloudflare", "huggingface", "openai"]
    assert status["configured_providers"][1]["provider"] == "cerebras"
    assert status["configured_providers"][2]["provider"] == "cloudflare"
    assert status["configured_providers"][2]["primary_model"] == "@cf/meta/llama-3.1-8b-instruct-fast"
    assert status["configured_providers"][3]["provider"] == "huggingface"


@pytest.mark.asyncio
async def test_llm_client_status_marks_provider_rate_limited_and_sets_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = [
        _MockResponse(
            status_code=429,
            payload={"error": {"message": "rate limit"}},
            text="rate limit",
            headers={"Retry-After": "120"},
        ),
        _MockResponse(
            status_code=429,
            payload={"error": {"message": "rate limit"}},
            text="rate limit",
            headers={"Retry-After": "120"},
        ),
        _MockResponse(
            status_code=200,
            payload={
                "id": "groq-1",
                "model": "llama-3.3-70b-versatile",
                "choices": [{"message": {"content": "fallback success"}}],
            },
        ),
    ]
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_mod.httpx,
        "AsyncClient",
        lambda **kwargs: _MockAsyncClient(queue=queue, calls=calls, **kwargs),
    )

    client = llm_mod.build_default_llm_client(
        llm_api_key="",
        llm_model="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_timeout_seconds=10.0,
        llm_max_output_tokens=200,
        llm_organization=None,
        llm_project=None,
        llm_provider="auto",
        gemini_api_key="gem-key",
        gemini_models=("gemini-2.0-flash",),
        groq_api_key="groq-key",
        groq_models=("llama-3.3-70b-versatile",),
    )

    result = await client.generate_text(system_prompt="sys", user_prompt="user")
    status = client.status()
    gemini_status = next(item for item in status["provider_statuses"] if item["provider"] == "gemini")
    groq_status = next(item for item in status["provider_statuses"] if item["provider"] == "groq")

    assert result is not None
    assert gemini_status["state"] == "rate_limited"
    assert gemini_status["last_status_code"] == 429
    assert gemini_status["cooldown_until"] is not None
    assert groq_status["state"] == "ok"


@pytest.mark.asyncio
async def test_llm_client_builds_cloudflare_openai_compatible_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    queue = [
        _MockResponse(
            status_code=200,
            payload={
                "id": "cf-1",
                "model": "@cf/meta/llama-3.1-8b-instruct-fast",
                "choices": [{"message": {"content": "cloudflare ok"}}],
            },
        )
    ]
    calls: list[dict] = []
    monkeypatch.setattr(
        llm_mod.httpx,
        "AsyncClient",
        lambda **kwargs: _MockAsyncClient(queue=queue, calls=calls, **kwargs),
    )

    client = llm_mod.build_default_llm_client(
        llm_api_key="",
        llm_model="gpt-4o-mini",
        llm_base_url="https://api.openai.com/v1",
        llm_timeout_seconds=10.0,
        llm_max_output_tokens=200,
        llm_organization=None,
        llm_project=None,
        llm_provider="cloudflare",
        cloudflare_account_id="acc-123",
        cloudflare_api_token="cf-token",
        cloudflare_models=("@cf/meta/llama-3.1-8b-instruct-fast",),
    )

    result = await client.generate_text(system_prompt="sys", user_prompt="user")

    assert result is not None
    assert result.text == "cloudflare ok"
    assert calls[0]["url"] == "https://api.cloudflare.com/client/v4/accounts/acc-123/ai/v1/chat/completions"
    assert calls[0]["headers"]["Authorization"] == "Bearer cf-token"
