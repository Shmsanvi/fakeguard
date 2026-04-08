# backend/pipelines/text_pipeline.py
"""
Text pipeline for FakeGuard.

Steps:
  1. Load fine-tuned RoBERTa classifier
  2. Compute metadata signals (source credibility, sentiment mismatch, NER consistency)
  3. Late-fuse RoBERTa logit + metadata vector → single text_score
"""

import torch
import numpy as np
import spacy
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from pathlib import Path

from backend.schemas import TextAnalysis
from backend.config import settings

logger = logging.getLogger(__name__)

# ─── Lazy-loaded globals ────────────────────────────────────────────────────
_roberta_model = None
_roberta_tokenizer = None
_sbert_model = None
_nlp = None          # spaCy NER


def _load_models():
    global _roberta_model, _roberta_tokenizer, _sbert_model, _nlp

    model_path = settings.TEXT_MODEL_PATH
    if Path(model_path).exists():
        logger.info(f"Loading fine-tuned RoBERTa from {model_path}")
        _roberta_tokenizer = RobertaTokenizer.from_pretrained(model_path)
        _roberta_model = RobertaForSequenceClassification.from_pretrained(model_path)
    else:
        # First run — use base model (will need fine-tuning)
        logger.warning("Fine-tuned model not found. Loading roberta-base. Run train_text.py first.")
        _roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        _roberta_model = RobertaForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        )

    _roberta_model.eval()

    _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
    _nlp = spacy.load("en_core_web_sm")


# ─── Public API ─────────────────────────────────────────────────────────────

def analyze_text(
    headline: str,
    body: str,
    source_credibility: float = 0.5,
) -> TextAnalysis:
    """
    Run the full text pipeline.

    Args:
        headline: Article headline string.
        body: Article body string (truncated internally to 512 tokens).
        source_credibility: Pre-computed credibility score from scraper (0–1).

    Returns:
        TextAnalysis with all scores and top influential tokens.
    """
    if _roberta_model is None:
        _load_models()

    # 1. RoBERTa score
    roberta_score = _roberta_score(headline, body)

    # 2. Metadata signals
    sentiment_mismatch = _sentiment_mismatch(headline, body)
    ner_consistency = _ner_consistency(headline, body)
    metadata_score = _compute_metadata_score(
        source_credibility, sentiment_mismatch, ner_consistency
    )

    # 3. Fuse: 70% RoBERTa + 30% metadata
    combined = 0.70 * roberta_score + 0.30 * metadata_score

    # 4. Influential tokens (simple gradient-based approximation)
    top_tokens = _get_top_tokens(headline, body, n=8)

    return TextAnalysis(
        roberta_score=round(roberta_score, 4),
        metadata_score=round(metadata_score, 4),
        combined_score=round(combined, 4),
        source_credibility=round(source_credibility, 4),
        sentiment_mismatch=round(sentiment_mismatch, 4),
        ner_consistency=round(ner_consistency, 4),
        top_tokens=top_tokens,
    )


# ─── Internal helpers ────────────────────────────────────────────────────────

def _roberta_score(headline: str, body: str) -> float:
    """
    Returns probability of FAKE (class 1) from fine-tuned RoBERTa.
    Input is headline + [SEP] + first 400 chars of body, truncated to 512 tokens.
    """
    text = f"{headline} </s> {body[:2000]}"
    inputs = _roberta_tokenizer(
        text,
        return_tensors="pt",
        max_length=512,
        truncation=True,
        padding=True,
    )
    with torch.no_grad():
        logits = _roberta_model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0][1])   # prob of FAKE label


def _sentiment_mismatch(headline: str, body: str) -> float:
    """
    Cosine distance between SBERT embeddings of headline and body.
    High distance = mismatch = suspicious.
    Returns 0 (aligned) to 1 (mismatched).
    """
    emb_h = _sbert_model.encode([headline])
    emb_b = _sbert_model.encode([body[:500]])
    sim = cosine_similarity(emb_h, emb_b)[0][0]
    # sim is in [-1, 1]; convert distance to [0, 1]
    return float(np.clip(1 - sim, 0, 1))


def _ner_consistency(headline: str, body: str) -> float:
    """
    Fraction of named entities in headline that also appear in body.
    Low overlap = inconsistency = suspicious.
    Returns 0 (inconsistent) to 1 (consistent).
    Score is INVERTED before use: 0=good, 1=bad.
    """
    doc_h = _nlp(headline)
    doc_b = _nlp(body[:2000])

    h_entities = {ent.text.lower() for ent in doc_h.ents}
    b_entities = {ent.text.lower() for ent in doc_b.ents}

    if not h_entities:
        return 0.5   # no entities in headline → neutral

    overlap = h_entities & b_entities
    consistency_ratio = len(overlap) / len(h_entities)
    inconsistency = 1.0 - consistency_ratio
    return float(inconsistency)


def _compute_metadata_score(
    source_credibility: float,
    sentiment_mismatch: float,
    ner_consistency: float,
) -> float:
    """
    Weighted combination of three metadata signals into a single [0,1] fake score.
    source_credibility is already 0=suspicious / 1=credible → invert it.
    """
    inverted_credibility = 1.0 - source_credibility
    score = (
        0.40 * inverted_credibility
        + 0.35 * sentiment_mismatch
        + 0.25 * ner_consistency
    )
    return float(np.clip(score, 0, 1))


def _get_top_tokens(headline: str, body: str, n: int = 8) -> list[str]:
    """
    Return the n most influential tokens using input × gradient saliency.
    Falls back to TF-IDF-style word frequency if gradients aren't available.
    """
    text = f"{headline} {body[:1000]}"
    inputs = _roberta_tokenizer(
        text, return_tensors="pt", max_length=512, truncation=True
    )

    try:
        _roberta_model.zero_grad()
        embeddings = _roberta_model.roberta.embeddings(inputs["input_ids"])
        embeddings.retain_grad()

        logits = _roberta_model(inputs_embeds=embeddings).logits
        fake_prob = torch.softmax(logits, dim=-1)[0][1]
        fake_prob.backward()

        saliency = embeddings.grad.abs().sum(dim=-1).squeeze()
        token_ids = inputs["input_ids"].squeeze().tolist()
        tokens = _roberta_tokenizer.convert_ids_to_tokens(token_ids)

        # Filter special tokens and subword prefixes
        scored = [
            (tok.lstrip("Ġ"), float(sal))
            for tok, sal in zip(tokens, saliency.tolist())
            if tok not in ["<s>", "</s>", "<pad>"] and len(tok.lstrip("Ġ")) > 2
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [tok for tok, _ in scored[:n]]

    except Exception as e:
        logger.warning(f"Gradient saliency failed: {e}. Using fallback.")
        words = text.split()
        freq = {}
        for w in words:
            w_clean = w.lower().strip(".,!?\"'")
            if len(w_clean) > 3:
                freq[w_clean] = freq.get(w_clean, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:n]
