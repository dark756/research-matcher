"""bge-m3 via FlagEmbedding: dense (1024-d, L2-normalised) + sparse lexical weights. GPU for batch stages, CPU for queries."""
from __future__ import annotations

import gc
from typing import Any

import numpy as np

from ..config import load_settings


class BGE:
    def __init__(self, device: str = "cuda"):
        from FlagEmbedding import BGEM3FlagModel

        self.device = device
        self.model = BGEM3FlagModel(load_settings().models["embed"], use_fp16=(device != "cpu"), devices=device)

    def encode(self, texts: list[str], max_length: int = 512, batch_size: int = 32,
               sparse: bool = True) -> tuple[np.ndarray, list[dict[str, float]] | None]:
        out: dict[str, Any] = self.model.encode(texts, batch_size=batch_size, max_length=max_length,
                                                return_dense=True, return_sparse=sparse, return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        lex = [{str(k): float(v) for k, v in d.items()} for d in out["lexical_weights"]] if sparse else None
        return dense, lex

    def close(self) -> None:
        """Free VRAM before gemma3 runs (plan §10.3)."""
        del self.model
        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def sparse_score(a: dict[str, float], b: dict[str, float]) -> float:
    """bge-m3 lexical matching score: sum of weight products over shared tokens."""
    if len(a) > len(b):
        a, b = b, a
    return float(sum(w * b[t] for t, w in a.items() if t in b))
