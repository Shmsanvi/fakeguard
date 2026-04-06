from pydantic import BaseModel, HttpUrl
from typing import Optional
from enum import Enum

class Verdict(str, Enum):
    REAL = "REAL"
    FAKE = "FAKE"
    UNCERTAIN = "UNCERTAIN"

class TextAnalysis(BaseModel):
    roberta_score: float
    metadata_score: float
    combined_score: float
    source_credibility: float
    sentiment_mismatch: float
    ner_consistency: float
    top_tokens: list[str]

class ImageAnalysis(BaseModel):
    ela_score: float
    frequency_score: float
    efficientnet_score: float
    combined_score: float
    ela_heatmap_b64: Optional[str] = None

class FusionResult(BaseModel):
    text_score: float
    image_score: float
    consistency_score: float
    final_score: float
    verdict: Verdict
    confidence_pct: float

class AnalysisResponse(BaseModel):
    verdict: Verdict
    confidence_pct: float
    final_score: float
    text_analysis: Optional[TextAnalysis] = None
    image_analysis: Optional[ImageAnalysis] = None
    fusion: FusionResult
    processing_time_ms: int
    warnings: list[str] = []

class URLRequest(BaseModel):
    url: HttpUrl
    include_heatmap: bool = False

class TextImageRequest(BaseModel):
    text: str
    image_url: Optional[str] = None
    include_heatmap: bool = False