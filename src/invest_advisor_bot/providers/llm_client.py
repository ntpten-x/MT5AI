from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import httpx
from loguru import logger


@dataclass(slots=True, frozen=True)
class LLMTextResponse:
    text: str
    model: str
    response_id: str | None
    raw_response: dict[str, Any]


class OpenAILLMClient:
    """Async wrapper for the OpenAI Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "gpt-5-mini",
        base_url: str = "https://api.openai.com/v1",
        timeout: float = 30.0,
        max_output_tokens: int = 1_000,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
        self.api_key = api_key.strip()
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.organization = organization
        self.project = project

    async def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        metadata: Mapping[str, str] | None = None,
    ) -> LLMTextResponse | None:
        if not self.api_key:
            logger.warning("OpenAI API key is missing; skipping LLM request")
            return None

        payload: dict[str, Any] = {
            "model": model or self.model,
            "instructions": system_prompt,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": user_prompt,
                        }
                    ],
                }
            ],
            "max_output_tokens": self.max_output_tokens,
        }
        if metadata:
            payload["metadata"] = dict(metadata)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.organization:
            headers["OpenAI-Organization"] = self.organization
        if self.project:
            headers["OpenAI-Project"] = self.project

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/responses",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "OpenAI Responses API returned HTTP {}: {}",
                exc.response.status_code,
                exc.response.text,
            )
            return None
        except httpx.HTTPError as exc:
            logger.warning("OpenAI Responses API request failed: {}", exc)
            return None

        try:
            response_json = response.json()
        except ValueError as exc:
            logger.warning("Failed to decode OpenAI response JSON: {}", exc)
            return None

        text = self._extract_output_text(response_json)
        if not text:
            logger.warning("OpenAI response did not include text output")
            return None

        return LLMTextResponse(
            text=text,
            model=str(response_json.get("model") or payload["model"]),
            response_id=self._as_optional_str(response_json.get("id")),
            raw_response=response_json,
        )

    @staticmethod
    def _extract_output_text(response_json: Mapping[str, Any]) -> str:
        direct_output = response_json.get("output_text")
        if isinstance(direct_output, str) and direct_output.strip():
            return direct_output.strip()

        output = response_json.get("output", [])
        fragments: list[str] = []
        if isinstance(output, list):
            for item in output:
                if not isinstance(item, Mapping):
                    continue
                content = item.get("content", [])
                if not isinstance(content, list):
                    continue
                for content_item in content:
                    if not isinstance(content_item, Mapping):
                        continue
                    text = content_item.get("text")
                    if isinstance(text, str) and text.strip():
                        fragments.append(text.strip())
                        continue
                    if isinstance(text, Mapping):
                        nested_text = text.get("value")
                        if isinstance(nested_text, str) and nested_text.strip():
                            fragments.append(nested_text.strip())

        return "\n".join(fragments).strip()

    @staticmethod
    def _as_optional_str(value: object) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None
