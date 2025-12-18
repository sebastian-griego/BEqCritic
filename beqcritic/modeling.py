"""
Cross-encoder equivalence scorer built on HuggingFace Transformers.

The model encodes text pairs (A, B) and outputs P(equivalent).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

def _load_calibrated_temperature(model_path: str) -> float | None:
    p = Path(model_path)
    if not p.exists() or not p.is_dir():
        return None
    for fname in ["temperature.json", "beqcritic_temperature.json"]:
        fp = p / fname
        if not fp.exists():
            continue
        try:
            obj = json.loads(fp.read_text(encoding="utf-8"))
            t = float(obj.get("temperature"))
            if t > 0:
                return float(t)
        except Exception:
            continue
    return None

@dataclass
class BeqCritic:
    model_name_or_path: str
    max_length: int = 512
    device: str | None = None
    temperature: float | None = None

    def __post_init__(self):
        model_path = _resolve_model_name_or_path(self.model_name_or_path)
        local_only = Path(model_path).exists()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=local_only)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            local_files_only=local_only,
        )
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()
        if self.temperature is None:
            self.temperature = _load_calibrated_temperature(model_path) or 1.0
        self.temperature = float(self.temperature)
        if self.temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {self.temperature!r}")

    @torch.inference_mode()
    def score_pairs(
        self,
        pairs: list[tuple[str, str]],
        batch_size: int = 16,
        temperature: float | None = None,
    ) -> list[float]:
        scores: list[float] = []
        temp = float(self.temperature) if temperature is None else float(temperature)
        if temp <= 0:
            raise ValueError(f"temperature must be > 0, got {temp!r}")
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i+batch_size]
            a = [x[0] for x in chunk]
            b = [x[1] for x in chunk]
            enc = self.tokenizer(
                a, b,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            if temp != 1.0:
                logits = logits / temp
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            scores.extend(probs)
        return scores
