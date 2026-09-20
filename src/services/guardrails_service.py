"""
Faithfulness and Hallucination Guardrails Service.
Evaluates RAG Triad metrics (Faithfulness, Context Relevance, Grounding)
to ensure synthesized answers are strictly backed by source citations.
"""

import re
from typing import List
from src.models.schema import Citation, GuardrailEvaluation


class GuardrailsService:
    """
    Evaluates grounding and hallucination metrics for RAG synthesized answers.
    """

    def __init__(self, faithfulness_threshold: float = 0.70) -> None:
        self.faithfulness_threshold = faithfulness_threshold
        self.stopwords = {
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "in", "on", "at", "to", "for", "of", "with", "by", "from", "as",
            "this", "that", "it", "its", "be", "been", "being", "have", "has"
        }

    def _extract_keywords(self, text: str) -> set[str]:
        """
        Extracts lowercase content words excluding stop words.
        """
        words = re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower())
        return {w for w in words if len(w) >= 2 and w not in self.stopwords}

    def evaluate_answer(
        self,
        question: str,
        answer: str,
        citations: List[Citation],
    ) -> GuardrailEvaluation:
        """
        Evaluates faithfulness score, context relevance score, and verification notes.
        """
        if not citations or not answer.strip():
            return GuardrailEvaluation(
                faithfulness_score=0.0,
                context_relevance_score=0.0,
                is_grounded=False,
                verification_notes=["No citation context available to verify claims."],
            )

        # 1. Split answer into individual claim clauses or sentences
        cleaned_answer = re.sub(r"^Based on .*?:\s*", "", answer).strip()
        raw_claims = re.split(r"(?<=[.!?])\s+", cleaned_answer)
        claims = [c.strip() for c in raw_claims if len(c.strip()) > 5]

        if not claims:
            claims = [cleaned_answer]

        # Combine all citation snippets into single corpus
        all_citation_text = " ".join([c.snippet for c in citations])
        citation_keywords = self._extract_keywords(all_citation_text)

        # 2. Evaluate claim-by-claim faithfulness
        verified_count = 0
        notes: List[str] = []

        for idx, claim in enumerate(claims, start=1):
            claim_keywords = self._extract_keywords(claim)
            if not claim_keywords:
                verified_count += 1
                notes.append(f"Claim #{idx}: verified (trivial clause)")
                continue

            # Check overlap against citation keywords
            overlap = claim_keywords.intersection(citation_keywords)
            ratio = len(overlap) / float(len(claim_keywords))

            if ratio >= 0.40:
                verified_count += 1
                matched_str = ", ".join(list(overlap)[:3])
                notes.append(
                    f"Claim #{idx}: verified ({round(ratio * 100)}% keyword match: {matched_str})"
                )
            else:
                notes.append(
                    f"Claim #{idx}: potentially ungrounded claim ({round(ratio * 100)}% match)"
                )

        faithfulness = round(verified_count / float(max(1, len(claims))), 2)

        # 3. Evaluate context relevance: question keywords matching citation snippets
        question_keywords = self._extract_keywords(question)
        if question_keywords:
            q_overlap = question_keywords.intersection(citation_keywords)
            context_relevance = round(min(1.0, len(q_overlap) / float(len(question_keywords))), 2)
        else:
            context_relevance = 1.0

        # Grounding flag: passes threshold
        is_grounded = bool(faithfulness >= self.faithfulness_threshold and len(citations) > 0)

        return GuardrailEvaluation(
            faithfulness_score=faithfulness,
            context_relevance_score=context_relevance,
            is_grounded=is_grounded,
            verification_notes=notes,
        )


_guardrails_instance = GuardrailsService()


def get_guardrails_service() -> GuardrailsService:
    """
    Returns the singleton guardrails evaluator service.
    """
    global _guardrails_instance
    return _guardrails_instance
