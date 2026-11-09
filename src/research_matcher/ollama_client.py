"""Thin Ollama client: structured chat, unload, health."""
from __future__ import annotations

import json
from typing import Any

import httpx

from .config import load_settings


class OllamaClient:
    def __init__(self, url: str | None = None, model: str | None = None):
        s = load_settings()
        self.url = (url or s.models["ollama_url"]).rstrip("/")
        self.model = model or s.models["llm"]
        self.http = httpx.Client(timeout=600)
        self.last: dict = {}

    def health(self) -> dict[str, Any]:
        """Returns {ok, model_present, capabilities, details}."""
        try:
            tags = self.http.get(f"{self.url}/api/tags", timeout=5).json()
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}
        names = [m["name"] for m in tags.get("models", [])]
        present = self.model in names
        out: dict[str, Any] = {"ok": present, "model_present": present, "models": names}
        if present:
            try:
                show = self.http.post(f"{self.url}/api/show", json={"model": self.model}, timeout=10).json()
                out["capabilities"] = show.get("capabilities", [])
            except Exception:  # noqa: BLE001
                pass
        return out

    def chat(self, messages: list[dict], schema: dict | None = None, options: dict | None = None,
             keep_alive: str = "30m") -> str:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False,
                                   "keep_alive": keep_alive, "options": options or {}}
        if schema is not None:
            payload["format"] = schema
        r = self.http.post(f"{self.url}/api/chat", json=payload)
        r.raise_for_status()
        self.last = r.json()   # usage: prompt_eval_count, eval_count, done_reason, ...
        return self.last["message"]["content"]

    def chat_json(self, messages: list[dict], schema: dict, options: dict | None = None) -> Any:
        return json.loads(self.chat(messages, schema, options))

    def unload(self) -> None:
        self.http.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=30)
