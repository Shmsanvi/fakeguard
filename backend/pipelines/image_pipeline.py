# backend/pipelines/image_pipeline.py
"""
Image pipeline for FakeGuard.

Three parallel detectors:
  1. ELA  — Error Level Analysis (detects spliced/edited regions)
  2. FFT  — Frequency analysis (detects GAN checkerboard / upsampling artifacts)
  3. CNN  — EfficientNet-B0 fine-tuned on CIFAKE + FaceForensics++

Each returns a score in [0, 1] where 1 = likely manipulated/AI-generated.
Final image_score = weighted ensemble of all three.
"""

import cv2
import numpy as np
import torch
import timm
import torchvision.transforms as T
from PIL import Image, ImageFilter
import io
import base64
import logging
from pathlib import Path
from scipy import fft as scipy_fft

from backend.schemas import ImageAnalysis
from backend.config import settings

logger = logging.getLogger(__name__)

# ─── Lazy-loaded globals ─────────────────────────────────────────────────────
_efficientnet_model = None
_img_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_efficientnet():
    global _efficientnet_model
    model_path = settings.IMAGE_MODEL_PATH

    if Path(model_path).exists():
        logger.info(f"Loading fine-tuned EfficientNet from {model_path}")
        _efficientnet_model = timm.create_model("efficientnet_b0", num_classes=2)
        state = torch.load(model_path, map_location="cpu")
        _efficientnet_model.load_state_dict(state)
    else:
        logger.warning("Fine-tuned image model not found. Using ImageNet weights only.")
        _efficientnet_model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=2
        )

    _efficientnet_model.eval()


# ─── Public API ──────────────────────────────────────────────────────────────

def analyze_image(
    image_bytes: bytes,
    include_heatmap: bool = False,
) -> ImageAnalysis:
    """
    Run all three detection methods and ensemble their scores.

    Args:
        image_bytes: Raw image bytes (JPEG/PNG).
        include_heatmap: If True, attach base64 ELA heatmap PNG.

    Returns:
        ImageAnalysis with per-detector scores and combined score.
    """
    if _efficientnet_model is None:
        _load_efficientnet()

    img_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(img_pil)

    ela_score, ela_heatmap = _run_ela(image_bytes, img_pil)
    freq_score = _run_frequency_analysis(img_np)
    cnn_score = _run_efficientnet(img_pil)

    # Ensemble: EfficientNet is most reliable, others are supporting signals
    combined = (
        0.50 * cnn_score
        + 0.30 * ela_score
        + 0.20 * freq_score
    )

    heatmap_b64 = None
    if include_heatmap and ela_heatmap is not None:
        heatmap_b64 = _encode_heatmap(ela_heatmap)

    return ImageAnalysis(
        ela_score=round(ela_score, 4),
        frequency_score=round(freq_score, 4),
        efficientnet_score=round(cnn_score, 4),
        combined_score=round(combined, 4),
        ela_heatmap_b64=heatmap_b64,
    )


# ─── Detector 1: ELA ─────────────────────────────────────────────────────────

def _run_ela(image_bytes: bytes, img_pil: Image.Image):
    """
    Error Level Analysis:
    1. Re-save original at a known JPEG quality.
    2. Compute pixel-wise absolute difference.
    3. Amplify differences — edited regions show higher error levels.
    Returns (score, heatmap_array).
    """
    quality = settings.ELA_QUALITY
    scale = settings.ELA_SCALE

    # Re-compress at known quality
    buffer = io.BytesIO()
    img_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    recompressed = Image.open(buffer).convert("RGB")

    # Compute difference
    orig_arr = np.array(img_pil).astype(np.float32)
    recomp_arr = np.array(recompressed).astype(np.float32)
    diff = np.abs(orig_arr - recomp_arr) * scale

    # Heatmap: mean across channels, normalised to [0, 255]
    heatmap = diff.mean(axis=2)
    heatmap_norm = np.clip(heatmap, 0, 255).astype(np.uint8)

    # Score: proportion of pixels with high error (above 80th percentile)
    threshold = np.percentile(heatmap_norm, 80)
    high_error_ratio = float((heatmap_norm > threshold).mean())

    # Normalise to a suspicion score: we expect some noise, flag if extreme
    score = float(np.clip(high_error_ratio * 3.0, 0, 1))

    return score, heatmap_norm


# ─── Detector 2: Frequency analysis ──────────────────────────────────────────

def _run_frequency_analysis(img_np: np.ndarray) -> float:
    """
    FFT-based GAN artifact detector.

    GANs trained with transposed convolutions leave a characteristic
    checkerboard pattern in the frequency domain — a grid of peaks
    that doesn't appear in real camera photos.

    Method:
    1. Convert to grayscale.
    2. Apply 2D FFT, shift DC to centre.
    3. Look for non-DC spectral peaks arranged in a regular grid.
    Returns score in [0, 1].
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Hanning window to suppress edge ringing
    h, w = gray.shape
    window = np.outer(np.hanning(h), np.hanning(w))
    gray_windowed = gray * window

    # 2D FFT magnitude spectrum (log scale)
    f = scipy_fft.fft2(gray_windowed)
    f_shift = scipy_fft.fftshift(f)
    magnitude = np.log1p(np.abs(f_shift))

    # Normalise
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

    # Mask out DC component (centre 5x5)
    cy, cx = h // 2, w // 2
    magnitude[cy-3:cy+3, cx-3:cx+3] = 0

    # Find peaks: pixels above 95th percentile
    threshold = np.percentile(magnitude, 95)
    peaks = magnitude > threshold
    peak_coords = np.column_stack(np.where(peaks))

    if len(peak_coords) < 4:
        return 0.0   # not enough peaks to form a grid pattern

    # Regularity check: compute pairwise distances between peaks
    # A GAN checkerboard creates peaks at regular intervals
    from scipy.spatial.distance import pdist
    distances = pdist(peak_coords)
    if len(distances) == 0:
        return 0.0

    # High regularity (low coefficient of variation) = suspicious
    mean_dist = distances.mean()
    std_dist = distances.std()
    cv = std_dist / (mean_dist + 1e-8)  # coefficient of variation

    # Low CV = peaks are regularly spaced = GAN artifact signature
    score = float(np.clip(1.0 - cv * 2.0, 0, 1))

    # Combine with peak density signal
    peak_density = float(peaks.mean()) * 20   # scale up for sensitivity
    score = np.clip(0.6 * score + 0.4 * peak_density, 0, 1)

    return float(score)


# ─── Detector 3: EfficientNet-B0 ─────────────────────────────────────────────

def _run_efficientnet(img_pil: Image.Image) -> float:
    """
    Run fine-tuned EfficientNet-B0.
    Returns probability that image is AI-generated / manipulated.
    """
    tensor = _img_transform(img_pil).unsqueeze(0)  # (1, 3, 224, 224)
    with torch.no_grad():
        logits = _efficientnet_model(tensor)
    probs = torch.softmax(logits, dim=-1)
    return float(probs[0][1])   # prob of FAKE class


# ─── Utility ──────────────────────────────────────────────────────────────────

def _encode_heatmap(heatmap: np.ndarray) -> str:
    """Convert grayscale heatmap to base64-encoded colourised PNG."""
    coloured = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    _, buf = cv2.imencode(".png", coloured)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def load_image_from_url(url: str, timeout: int = 20) -> bytes:
    """Download image bytes from a URL."""
    import requests
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FakeGuard/1.0)"}
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.content