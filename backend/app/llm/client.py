from __future__ import annotations

from typing import Dict, Iterator, List, Optional


class LLMClient:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        from openai import OpenAI

        self.model = model
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self, messages: List[Dict[str, str]], temperature: float = 0.1, **kwargs
    ) -> str:
        resp = self._client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, **kwargs
        )
        return resp.choices[0].message.content or ""

    def stream_chat(
        self, messages: List[Dict[str, str]], temperature: float = 0.1, **kwargs
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
