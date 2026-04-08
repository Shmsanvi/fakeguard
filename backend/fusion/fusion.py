# backend/fusion/fusion.py
"""
Multimodal Fusion Layer for FakeGuard.

Combines text_score + image_score and cross-checks
whether the image is semantically consistent with the article
using OpenAI CLIP embeddings.

Final pipeline:
  text_score   (0=real, 1=fake)  ─┐
  image_score  (0=real, 1=fake)  ─┼─► weighted_score ─┐
  consistency  (0=mismatch,1=ok) ─┘                    ├─► final_score → verdict
                                  penalty if mismatch ──┘
"""

import numpy as np
import torch
import clip
from PIL import Image
import io
import logging
from typing import Optional

from backend.schemas import FusionResult, Verdict
from backend.config import settings

logger = logging.getLogger(__name__)

_clip_model = None
_clip_preprocess = None
_clip_device = "cpu"


def _load_clip():
    global _clip_model, _clip_preprocess, _clip_device
    _clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=_clip_device)
    _clip_model.eval()
    logger.info(f"CLIP loaded on {_clip_device}")


# ─── Public API ───────────────────────────────────────────────────────────────

def fuse(
    text_score: Optional[float],
    image_score: Optional[float],
    article_text: str = "",
    image_bytes: Optional[bytes] = None,
) -> FusionResult:
    """
    Fuse text and image scores with cross-modal consistency penalty.

    Args:
        text_score:   float in [0,1], None if no text available.
        image_score:  float in [0,1], None if no image available.
        article_text: raw article text for CLIP text embedding.
        image_bytes:  raw image bytes for CLIP image embedding.

    Returns:
        FusionResult with final_score, verdict, and confidence.
    """
    if _clip_model is None:
        _load_clip()

    tw = settings.TEXT_WEIGHT
    iw = settings.IMAGE_WEIGHT

    # ── Step 1: weighted combination ──────────────────────────────────────────
    if text_score is not None and image_score is not None:
        weighted = tw * text_score + iw * image_score
    elif text_score is not None:
        weighted = text_score
    elif image_score is not None:
        weighted = image_score
    else:
        raise ValueError("At least one of text_score or image_score must be provided.")

    # ── Step 2: cross-modal consistency via CLIP ───────────────────────────────
    consistency = _compute_clip_consistency(article_text, image_bytes)

    # ── Step 3: apply consistency penalty ─────────────────────────────────────
    # Low consistency = image doesn't match article = suspicious
    # We boost the fake score when there's a big mismatch
    mismatch_penalty = _consistency_penalty(consistency)
    final_score = float(np.clip(weighted + mismatch_penalty, 0, 1))

    # ── Step 4: verdict + confidence ──────────────────────────────────────────
    verdict, confidence = _score_to_verdict(final_score)

    return FusionResult(
        text_score=round(text_score if text_score is not None else -1.0, 4),
        image_score=round(image_score if image_score is not None else -1.0, 4),
        consistency_score=round(consistency, 4),
        final_score=round(final_score, 4),
        verdict=verdict,
        confidence_pct=round(confidence, 1),
    )


# ─── CLIP consistency ─────────────────────────────────────────────────────────

def _compute_clip_consistency(text: str, image_bytes: Optional[bytes]) -> float:
    """
    Compute cosine similarity between CLIP text and image embeddings.
    Returns 0.0 if either is unavailable.
    """
    if not text or not image_bytes:
        return 0.5   # neutral — can't judge

    try:
        # Text embedding
        text_trunc = text[:200]   # CLIP token limit
        tokens = clip.tokenize([text_trunc], truncate=True).to(_clip_device)
        with torch.no_grad():
            text_emb = _clip_model.encode_text(tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Image embedding
        img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_tensor = _clip_preprocess(img_pil).unsqueeze(0).to(_clip_device)
        with torch.no_grad():
            img_emb = _clip_model.encode_image(img_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # Cosine similarity → [0, 1]
        sim = (text_emb @ img_emb.T).item()
        return float(np.clip((sim + 1) / 2, 0, 1))   # shift from [-1,1] to [0,1]

    except Exception as e:
        logger.warning(f"CLIP consistency check failed: {e}")
        return 0.5


def _consistency_penalty(consistency: float) -> float:
    """
    Convert consistency score to a fake-score penalty.
    Only penalise when there's a clear mismatch (below threshold).
    No reward for high consistency — that's already captured in individual scores.
    """
    threshold = settings.CONSISTENCY_THRESHOLD
    if consistency < threshold:
        # Linear penalty up to +0.15 for complete mismatch
        penalty = (threshold - consistency) / threshold * 0.15
        return float(penalty)
    return 0.0


# ─── Verdict mapping ──────────────────────────────────────────────────────────

def _score_to_verdict(score: float):
    """
    Map [0,1] fake score to Verdict enum + confidence percentage.
    """
    fake_thresh = settings.FAKE_THRESHOLD
    real_thresh = settings.REAL_THRESHOLD

    if score >= fake_thresh:
        verdict = Verdict.FAKE
        # Confidence scales from 50% at threshold to 99% at score=1.0
        confidence = 50 + 49 * (score - fake_thresh) / (1 - fake_thresh)
    elif score <= real_thresh:
        verdict = Verdict.REAL
        confidence = 50 + 49 * (real_thresh - score) / real_thresh
    else:
        verdict = Verdict.UNCERTAIN
        # Confidence = how far from the middle of the uncertain zone (0.5)
        dist_from_mid = abs(score - 0.5)
        confidence = 50 + 30 * (dist_from_mid / 0.10)

    return verdict, float(np.clip(confidence, 50, 99))
