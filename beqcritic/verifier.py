"""
NL->Lean verifier for reference-free reranking.

Scores (nl_statement, lean_statement) pairs; higher means "more likely correct".
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .features import extract_features
from .textnorm import normalize_lean_statement, normalize_whitespace

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve_model_name_or_path(name_or_path: str) -> str:
    p = Path(name_or_path)
    if p.exists():
        return name_or_path
    if not p.is_absolute():
        from_repo = REPO_ROOT / p
        if from_repo.exists():
            return str(from_repo)
    if "/" in name_or_path:
        local = REPO_ROOT / "hf_models" / name_or_path.replace("/", "--")
        if local.exists():
            return str(local)
    return name_or_path


@dataclass
class NLVerifier:
    model_name_or_path: str
    max_length: int = 512
    device: str | None = None
    use_features: bool = True

    def __post_init__(self) -> None:
        model_path = _resolve_model_name_or_path(self.model_name_or_path)
        local_only = Path(model_path).exists()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=local_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=local_only,
        )
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    def _prep(self, nl: str, lean: str) -> tuple[str, str]:
        nl_clean = normalize_whitespace(nl)
        lean_clean = normalize_lean_statement(lean)
        if self.use_features:
            feats = extract_features(lean_clean).to_prefix()
            lean_clean = f"{feats} {lean_clean}"
        return nl_clean, lean_clean

    @torch.inference_mode()
    def score_pairs(
        self,
        nl_list: list[str],
        lean_list: list[str],
        batch_size: int = 16,
    ) -> list[float]:
        if len(nl_list) != len(lean_list):
            raise ValueError(f"Length mismatch: nl={len(nl_list)} lean={len(lean_list)}")
        scores: list[float] = []
        for i in range(0, len(nl_list), batch_size):
            chunk_nl = nl_list[i : i + batch_size]
            chunk_lean = lean_list[i : i + batch_size]
            nl_clean = []
            lean_clean = []
            for n, l in zip(chunk_nl, chunk_lean):
                n2, l2 = self._prep(n, l)
                nl_clean.append(n2)
                lean_clean.append(l2)
            enc = self.tokenizer(
                nl_clean,
                lean_clean,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            if logits.dim() == 2 and logits.size(-1) == 1:
                vals = logits.squeeze(-1)
            elif logits.dim() == 2 and logits.size(-1) >= 2:
                vals = logits[:, 1]
            else:
                vals = logits.view(-1)
            scores.extend(vals.detach().cpu().tolist())
        return scores
