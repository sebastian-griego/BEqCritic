"""
Simple text embedder for kNN preselection.

This is used to avoid scoring all O(n^2) candidate pairs with the cross-encoder:
  1) embed each candidate once (bi-encoder style)
  2) build a kNN graph
  3) score only those edges with the cross-encoder critic

The embedder uses HuggingFace `AutoModel` with mean pooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .modeling import _resolve_model_name_or_path
from .textnorm import normalize_lean_statement


@dataclass
class TextEmbedder:
    model_name_or_path: str
    max_length: int = 256
    device: str | None = None

    def __post_init__(self) -> None:
        model_path = _resolve_model_name_or_path(self.model_name_or_path)
        local_only = Path(model_path).exists()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=local_only)
        self.model = AutoModel.from_pretrained(model_path, local_files_only=local_only)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed_statements(
        self,
        statements: list[str],
        batch_size: int = 32,
        normalize_statements: bool = True,
        return_cpu: bool = True,
    ) -> torch.Tensor:
        """
        Return L2-normalized embeddings with shape [n, d].
        """
        if not statements:
            raise ValueError("No statements provided")
        texts = [normalize_lean_statement(s) for s in statements] if normalize_statements else [str(s) for s in statements]
        outs: list[torch.Tensor] = []
        for i in range(0, len(texts), int(batch_size)):
            chunk = texts[i : i + int(batch_size)]
            enc = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=int(self.max_length),
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            h = out.last_hidden_state
            mask = enc["attention_mask"].to(h.dtype).unsqueeze(-1)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = F.normalize(pooled, dim=-1)
            outs.append(pooled.detach())
        emb = torch.cat(outs, dim=0)
        return emb.cpu() if return_cpu else emb
