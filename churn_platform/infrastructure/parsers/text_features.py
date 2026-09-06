"""Dependency-free keyword sentiment mining for support text.

Real tenants rarely supply a labelled sentiment column, so churn intent has to
be read out of free text: ticket subjects, complaint notes, dispute reasons.
This module scores text against a weighted lexicon of churn-intent terms.

It degrades gracefully. If a table has no free text, or the text is a repeated
boilerplate string with no lexicon hits, the score collapses to 0.0 for every
entity and simply stops differentiating — it never raises.
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable, NamedTuple

# Weights encode how strongly a term predicts churn. Explicit intent to leave
# scores highest, service-quality complaints lowest.
CHURN_LEXICON: dict[str, float] = {
    "cancel": 3.0,
    "unsubscribe": 3.0,
    "port out": 3.0,
    "mnp": 3.0,
    "switching": 2.5,
    "competitor": 2.5,
    "fraud": 2.5,
    "chargeback": 2.5,
    "refund": 2.0,
    "unresolved": 2.0,
    "escalat": 2.0,
    "not working": 2.0,
    "broken": 2.0,
    "terrible": 2.0,
    "unacceptable": 2.0,
    "unusable": 2.0,
    "overbilling": 2.0,
    "duplicate": 1.5,
    "too slow": 1.5,
    "slow": 1.5,
    "lag": 1.5,
    "expensive": 1.5,
    "cheaper": 1.5,
    "dropped": 1.5,
    "error": 1.5,
    "dispute": 1.5,
    "downgrade": 1.5,
    "leaving": 1.5,
}

DEFAULT_NEGATIVE_THRESHOLD = 2.0


class TextSentiment(NamedTuple):
    """Aggregate sentiment for one entity's support text."""

    score: float
    matched_terms: list[str]
    record_count: int
    is_negative: bool


class KeywordSentimentScorer:
    """Scores free text for churn intent using a weighted keyword lexicon."""

    def __init__(
        self,
        lexicon: dict[str, float] | None = None,
        negative_threshold: float = DEFAULT_NEGATIVE_THRESHOLD,
    ) -> None:
        self.lexicon = dict(lexicon) if lexicon is not None else dict(CHURN_LEXICON)
        self.negative_threshold = negative_threshold
        # Longest terms first so "port out" is matched before a shorter term
        # that happens to be a substring of it.
        self._terms = sorted(self.lexicon, key=len, reverse=True)

    def score_text(self, text: str) -> tuple[float, list[str]]:
        """Score a single string, returning (score, matched_terms)."""
        if not text:
            return 0.0, []
        lowered = str(text).lower()
        matched: list[str] = []
        total = 0.0
        # Terms are longest-first, so skipping any term that is a substring of an
        # already-matched one stops "too slow" from also scoring as "slow".
        for term in self._terms:
            if term not in lowered:
                continue
            if any(term in longer for longer in matched):
                continue
            matched.append(term)
            total += self.lexicon[term]
        return total, matched

    def score_texts(self, texts: Iterable[str]) -> TextSentiment:
        """Aggregate the sentiment of every support record belonging to one entity.

        The score is the mean per-record weight rather than the sum, so an entity
        with many mild tickets is not scored above one with a single explicit
        cancellation request.
        """
        materialized = [t for t in texts if t is not None and str(t).strip()]
        if not materialized:
            return TextSentiment(score=0.0, matched_terms=[], record_count=0, is_negative=False)

        total = 0.0
        term_counts: Counter[str] = Counter()
        for text in materialized:
            score, matched = self.score_text(text)
            total += score
            term_counts.update(matched)

        aggregate = total / len(materialized)
        ranked = [term for term, _ in term_counts.most_common()]
        return TextSentiment(
            score=round(aggregate, 4),
            matched_terms=ranked,
            record_count=len(materialized),
            is_negative=aggregate >= self.negative_threshold,
        )
