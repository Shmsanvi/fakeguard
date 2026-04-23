# backend/main.py
"""
FakeGuard FastAPI backend.

Endpoints:
  POST /analyze/url      — scrape article from URL, run full pipeline
  POST /analyze/upload   — raw text + optional image upload
  GET  /health           — service health check
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.schemas import AnalysisResponse, URLRequest, Verdict
from backend.config import settings
from backend.utils.scraper import scrape_article
from backend.pipelines.text_pipeline import analyze_text
from backend.pipelines.image_pipeline import analyze_image, load_image_from_url
from backend.fusion.fusion import fuse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── App lifespan (warm up models on startup) ────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Warming up models...")
    try:
        from backend.pipelines.text_pipeline import _load_models as load_text
        from backend.pipelines.image_pipeline import _load_efficientnet
        from backend.fusion.fusion import _load_clip
        load_text()
        _load_efficientnet()
        _load_clip()
        logger.info("All models loaded successfully.")
    except Exception as e:
        logger.warning(f"Model preload failed (will load on first request): {e}")
    yield


app = FastAPI(
    title="FakeGuard API",
    version="1.0.0",
    description="Real-time hybrid fake news and image manipulation detector.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
   allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}


@app.post("/analyze/url", response_model=AnalysisResponse)
async def analyze_url(request: URLRequest):
    """
    Accept a news article URL, scrape it, and run the full detection pipeline.
    """
    start = time.time()
    warnings = []
    url = str(request.url)

    # ── 1. Scrape ──
    try:
        article = scrape_article(url)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not scrape URL: {e}")

    if not article.body:
        raise HTTPException(status_code=422, detail="No article body found at this URL.")

    # ── 2. Text pipeline ──
    text_analysis = None
    text_score = None
    try:
        text_analysis = analyze_text(
            headline=article.headline,
            body=article.body,
            source_credibility=article.credibility_score,
        )
        text_score = text_analysis.combined_score
    except Exception as e:
        warnings.append(f"Text analysis failed: {e}")

    # ── 3. Image pipeline ──
    image_analysis = None
    image_score = None
    image_bytes = None
    if article.image_url:
        try:
            image_bytes = load_image_from_url(article.image_url)
            image_analysis = analyze_image(
                image_bytes=image_bytes,
                include_heatmap=request.include_heatmap,
            )
            image_score = image_analysis.combined_score
        except Exception as e:
            warnings.append(f"Image analysis failed: {e}")
    else:
        warnings.append("No image found in article — image analysis skipped.")

    if text_score is None and image_score is None:
        raise HTTPException(status_code=500, detail="Both pipelines failed.")

    # ── 4. Fusion ──
    fusion_result = fuse(
        text_score=text_score,
        image_score=image_score,
        article_text=article.body,
        image_bytes=image_bytes,
    )

    elapsed_ms = int((time.time() - start) * 1000)

    return AnalysisResponse(
        verdict=fusion_result.verdict,
        confidence_pct=fusion_result.confidence_pct,
        final_score=fusion_result.final_score,
        text_analysis=text_analysis,
        image_analysis=image_analysis,
        fusion=fusion_result,
        processing_time_ms=elapsed_ms,
        warnings=warnings,
    )


@app.post("/analyze/upload", response_model=AnalysisResponse)
async def analyze_upload(
    text: str = Form(...),
    headline: str = Form(default=""),
    image: Optional[UploadFile] = File(default=None),
    include_heatmap: bool = Form(default=False),
):
    """
    Accept raw text + optional image file upload and run the detection pipeline.
    Useful for manual testing or browser extension integration.
    """
    start = time.time()
    warnings = []

    # ── 1. Text pipeline ──
    text_analysis = None
    text_score = None
    try:
        text_analysis = analyze_text(
            headline=headline or text[:120],
            body=text,
            source_credibility=0.5,   # unknown source
        )
        text_score = text_analysis.combined_score
    except Exception as e:
        warnings.append(f"Text analysis failed: {e}")

    # ── 2. Image pipeline ──
    image_analysis = None
    image_score = None
    image_bytes = None
    if image is not None:
        size_bytes = 0
        image_bytes = await image.read()
        size_bytes = len(image_bytes)
        max_bytes = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024

        if size_bytes > max_bytes:
            warnings.append(f"Image too large ({size_bytes // 1024}KB). Skipping image analysis.")
            image_bytes = None
        else:
            try:
                image_analysis = analyze_image(
                    image_bytes=image_bytes,
                    include_heatmap=include_heatmap,
                )
                image_score = image_analysis.combined_score
            except Exception as e:
                warnings.append(f"Image analysis failed: {e}")

    if text_score is None and image_score is None:
        raise HTTPException(status_code=500, detail="Both pipelines failed.")

    # ── 3. Fusion ──
    fusion_result = fuse(
        text_score=text_score,
        image_score=image_score,
        article_text=text,
        image_bytes=image_bytes,
    )

    elapsed_ms = int((time.time() - start) * 1000)

    return AnalysisResponse(
        verdict=fusion_result.verdict,
        confidence_pct=fusion_result.confidence_pct,
        final_score=fusion_result.final_score,
        text_analysis=text_analysis,
        image_analysis=image_analysis,
        fusion=fusion_result,
        processing_time_ms=elapsed_ms,
        warnings=warnings,
    )


# ─── Run directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )