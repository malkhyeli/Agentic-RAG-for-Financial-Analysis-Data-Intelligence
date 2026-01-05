from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean, pstdev
from typing import Any, Sequence


@dataclass(frozen=True)
class RetrievalFeatures:
    top_score: float = 0.0
    mean_topk: float = 0.0
    score_gap: float = 0.0
    std_topk: float = 0.0
    num_quotes: int = 0
    unique_pages: int = 0
    page_spread: int = 0
    total_quote_chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_retrieval_features(evidence: Sequence[Any]) -> RetrievalFeatures:
    if not evidence:
        return RetrievalFeatures()

    scores = [float(getattr(e, "score", 0.0) or 0.0) for e in evidence]
    pages = [int(getattr(e, "page", 0) or 0) for e in evidence if int(getattr(e, "page", 0) or 0) > 0]
    quotes = [str(getattr(e, "quote", "") or "") for e in evidence]

    top = scores[0] if scores else 0.0
    mean_topk = mean(scores) if scores else 0.0
    std_topk = pstdev(scores) if len(scores) >= 2 else 0.0
    score_gap = (scores[0] - scores[1]) if len(scores) >= 2 else (scores[0] if scores else 0.0)

    uniq_pages = len(set(pages)) if pages else 0
    page_spread = (max(pages) - min(pages)) if len(pages) >= 2 else 0
    total_chars = sum(len(q) for q in quotes if q)

    return RetrievalFeatures(
        top_score=top,
        mean_topk=mean_topk,
        score_gap=score_gap,
        std_topk=std_topk,
        num_quotes=len(quotes),
        unique_pages=uniq_pages,
        page_spread=page_spread,
        total_quote_chars=total_chars,
    )

