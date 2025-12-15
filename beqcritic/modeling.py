"""
Cross-encoder equivalence scorer built on HuggingFace Transformers.

The model encodes text pairs (A, B) and outputs P(equivalent).
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

@dataclass
class BeqCritic:
    model_name_or_path: str
    max_length: int = 512
    device: str | None = None

    def __post_init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, num_labels=2)
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_pairs(self, pairs: list[tuple[str, str]], batch_size: int = 16) -> list[float]:
        scores: list[float] = []
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
            probs = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().tolist()
            scores.extend(probs)
        return scores
