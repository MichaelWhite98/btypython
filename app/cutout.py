from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


MAX_EDGE = 1600
MIN_COMPONENT_AREA_RATIO = 0.015
MAX_CANDIDATES = 3


@dataclass
class CutoutCandidate:
    display_name: str
    score: float
    bbox: list[int]
    area_ratio: float
    mask_path: Path
    cutout_path: Path
    thumbnail_path: Path
    source_type: str


def _load_image(image_path: Path) -> tuple[np.ndarray, Image.Image]:
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    scale = min(1.0, MAX_EDGE / max(width, height))
    if scale < 1.0:
        pil_image = pil_image.resize((int(width * scale), int(height * scale)), Image.Resampling.LANCZOS)

    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return bgr, pil_image


def _initial_grabcut_mask(bgr: np.ndarray) -> np.ndarray:
    height, width = bgr.shape[:2]
    margin_x = max(8, int(width * 0.08))
    margin_y = max(8, int(height * 0.08))
    rect = (margin_x, margin_y, max(1, width - 2 * margin_x), max(1, height - 2 * margin_y))

    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype("uint8")
    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = cv2.GaussianBlur(foreground, (5, 5), 0)
    _, foreground = cv2.threshold(foreground, 60, 255, cv2.THRESH_BINARY)
    return foreground


def _classify_source_type(width: int, height: int) -> str:
    ratio = width / max(height, 1)
    if ratio < 0.85:
        return "cup"
    if ratio > 1.35:
        return "plate"
    return "bowl"


def _extract_components(mask: np.ndarray) -> list[tuple[int, int, int, int, np.ndarray]]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    total_area = mask.shape[0] * mask.shape[1]
    min_area = int(total_area * MIN_COMPONENT_AREA_RATIO)
    components: list[tuple[int, int, int, int, np.ndarray]] = []

    for label_idx in range(1, num_labels):
        x = int(stats[label_idx, cv2.CC_STAT_LEFT])
        y = int(stats[label_idx, cv2.CC_STAT_TOP])
        width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
        height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
        area = int(stats[label_idx, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = np.where(labels == label_idx, 255, 0).astype("uint8")
        components.append((x, y, width, height, component_mask))

    if not components:
        height, width = mask.shape[:2]
        components.append((0, 0, width, height, mask.copy()))

    components.sort(key=lambda item: np.count_nonzero(item[4]), reverse=True)
    return components[:MAX_CANDIDATES]


def _save_candidate_files(
    rgb_image: Image.Image,
    component_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_dir: Path,
    item_id: str,
) -> tuple[Path, Path, Path]:
    x, y, width, height = bbox
    alpha = Image.fromarray(component_mask, mode="L")
    rgba = rgb_image.copy().convert("RGBA")
    rgba.putalpha(alpha)

    mask_path = output_dir / f"{item_id}-mask.png"
    cutout_path = output_dir / f"{item_id}-cutout.png"
    thumbnail_path = output_dir / f"{item_id}-thumb.png"

    alpha.save(mask_path)
    cropped = rgba.crop((x, y, x + width, y + height))
    cropped.save(cutout_path)

    thumb = cropped.copy()
    thumb.thumbnail((320, 320), Image.Resampling.LANCZOS)
    thumb.save(thumbnail_path)

    return mask_path, cutout_path, thumbnail_path


def generate_cutout_candidates(image_path: Path, output_dir: Path, task_id: str) -> list[CutoutCandidate]:
    bgr, pil_image = _load_image(image_path)
    foreground_mask = _initial_grabcut_mask(bgr)
    components = _extract_components(foreground_mask)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_area = pil_image.size[0] * pil_image.size[1]
    candidates: list[CutoutCandidate] = []

    for index, (x, y, width, height, component_mask) in enumerate(components, start=1):
        item_id = f"{task_id}-item-{index}"
        mask_path, cutout_path, thumbnail_path = _save_candidate_files(
            pil_image,
            component_mask,
            (x, y, width, height),
            output_dir,
            item_id,
        )
        area = int(np.count_nonzero(component_mask))
        score = min(0.99, 0.55 + area / max(image_area, 1))
        candidates.append(
            CutoutCandidate(
                display_name=f"主体 {index}",
                score=round(float(score), 4),
                bbox=[x, y, x + width, y + height],
                area_ratio=round(area / max(image_area, 1), 4),
                mask_path=mask_path,
                cutout_path=cutout_path,
                thumbnail_path=thumbnail_path,
                source_type=_classify_source_type(width, height),
            )
        )

    return candidates


def candidate_to_dict(candidate: CutoutCandidate, base_url: str) -> dict[str, Any]:
    return {
        "id": candidate.cutout_path.stem.replace("-cutout", ""),
        "displayName": candidate.display_name,
        "score": candidate.score,
        "bbox": candidate.bbox,
        "areaRatio": candidate.area_ratio,
        "maskUrl": f"{base_url}/files/output/{candidate.mask_path.name}",
        "cutoutUrl": f"{base_url}/files/output/{candidate.cutout_path.name}",
        "thumbnailUrl": f"{base_url}/files/output/{candidate.thumbnail_path.name}",
        "sourceType": candidate.source_type,
    }
