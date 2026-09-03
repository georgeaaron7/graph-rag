"""embeddings with a local default (sentence-transformers) or an API backend.
"""
from __future__ import annotations

import threading
from typing import List, Optional


class Embedder:
    def __init__(
        self,
        backend: str = "local",
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.model = model
        self._api_base_url = api_base_url
        self._api_key = api_key
        self._st_model = None
        self._openai = None
        self._dim: Optional[int] = None
        self._lock = threading.Lock()

    def _ensure_local(self):
        if self._st_model is None:
            with self._lock:
                if self._st_model is None:
                    from sentence_transformers import SentenceTransformer

                    self._st_model = SentenceTransformer(self.model)
                    self._dim = self._st_model.get_sentence_embedding_dimension()

    def _ensure_api(self):
        if self._openai is None:
            from openai import OpenAI

            self._openai = OpenAI(base_url=self._api_base_url, api_key=self._api_key)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.backend == "local":
            self._ensure_local()
            vecs = self._st_model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            )
            return vecs.tolist()
        self._ensure_api()
        resp = self._openai.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

    @property
    def dimension(self) -> int:
        if self._dim is None:
            # infer lazily from a probe embedding (works for both backends)
            self._dim = len(self.embed_query("dimension probe"))
        return self._dim
