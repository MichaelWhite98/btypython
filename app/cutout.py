"""
BTY 抠图服务 - 核心算法模块

支持算法:
- rmbg: BRIA-RMBG-1.4 商业级模型（推荐，效果最好）
- birefnet: BiRefNet 最新模型（效果好）
- isnet: IS-Net 模型（效果好）
- u2net: U²-Net 通用模型（速度快）
- grabcut: OpenCV GrabCut 传统算法（最快，效果一般）
"""
from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from PIL import Image

# ============================================================================
# 配置
# ============================================================================

CUTOUT_CONFIG = {
    # 算法选择 (u2net 内存占用小，birefnet 内存占用大)
    'algorithm': 'u2net',

    # Rembg 模型映射
    'rembg_models': {
        'birefnet': 'birefnet-general',      # BiRefNet 效果好，但内存占用大
        'isnet': 'isnet-general-use',        # IS-Net 效果好
        'u2net': 'u2net',                    # U²-Net 通用，内存占用小
    },

    # GrabCut 配置
    'grabcut_iterations': 12,
    'grabcut_feather_width': 15,

    # 通用配置
    'max_edge': 512,  # 减小到 512 以降低内存占用
    'min_component_area_ratio': 0.01,
    'max_candidates': 3,
}

AlgorithmType = Literal['rmbg', 'birefnet', 'isnet', 'u2net', 'grabcut']

# Rembg session 缓存
_REMBG_SESSIONS = {}

# RMBG 模型缓存
_RMBG_MODEL = None
_RMBG_PROCESSOR = None
_RMBG_DEVICE = None  # 缓存设备信息


# ============================================================================
# 设备检测
# ============================================================================

def _get_device() -> str:
    """获取最佳可用设备 (cuda > cpu)"""
    import torch
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[GPU] Using GPU: {device_name}")
        return "cuda"
    print("[CPU] Using CPU (CUDA not available)")
    return "cpu"


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class CutoutCandidate:
    """抠图候选结果"""
    display_name: str
    score: float
    bbox: list[int]
    area_ratio: float
    mask_path: Path
    cutout_path: Path
    thumbnail_path: Path
    source_type: str
    algorithm: str


# ============================================================================
# Rembg 模型
# ============================================================================

def _get_rembg_session(model_name: str):
    """获取 Rembg session（缓存，自动使用 GPU）"""
    global _REMBG_SESSIONS

    if model_name not in _REMBG_SESSIONS:
        from rembg import new_session
        import torch

        # 检测是否有 GPU
        if torch.cuda.is_available():
            # 使用 CUDAExecutionProvider 加速
            print(f"🚀 Loading {model_name} with GPU acceleration")
            _REMBG_SESSIONS[model_name] = new_session(
                model_name,
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
        else:
            print(f"💻 Loading {model_name} on CPU")
            _REMBG_SESSIONS[model_name] = new_session(model_name)

    return _REMBG_SESSIONS[model_name]


def _rembg_remove_background(
    image_path: Path,
    model_name: str = 'birefnet-general',
    max_edge: int = 768
) -> tuple[Image.Image, np.ndarray]:
    """
    使用 Rembg 移除背景

    Args:
        image_path: 图片路径
        model_name: 模型名称
            - 'birefnet-general': BiRefNet 效果好
            - 'isnet-general-use': IS-Net 效果好
            - 'u2net': U²-Net 通用
        max_edge: 最大边长，用于缩放加速

    Returns:
        rgba: 带 alpha 通道的 RGBA 图像
        mask: mask 数组 (0-255)
    """
    from rembg import remove
    import time

    start_time = time.time()

    # 读取图片并缩小（减少内存占用）
    pil_image = Image.open(image_path).convert('RGB')
    orig_width, orig_height = pil_image.size

    # 缩放大图
    scale = min(1.0, max_edge / max(orig_width, orig_height))
    if scale < 1.0:
        pil_image = pil_image.resize(
            (int(orig_width * scale), int(orig_height * scale)),
            Image.Resampling.LANCZOS
        )
        print(f"[RESIZE] Scaled from {orig_width}x{orig_height} to {pil_image.size[0]}x{pil_image.size[1]}")

    # 转换为 bytes
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    input_bytes = buffer.getvalue()

    # 移除背景
    session = _get_rembg_session(model_name)
    output_bytes = remove(input_bytes, session=session)

    # 转换为 PIL Image
    rgba = Image.open(BytesIO(output_bytes)).convert('RGBA')

    # 恢复原始尺寸
    if scale < 1.0:
        rgba = rgba.resize((orig_width, orig_height), Image.Resampling.LANCZOS)

    # 提取 alpha 通道作为 mask
    alpha = rgba.split()[-1]
    mask = np.array(alpha)

    elapsed = time.time() - start_time
    print(f"[TIME] Rembg ({model_name}) inference time: {elapsed:.2f}s")

    return rgba, mask


# ============================================================================
# RMBG-1.4 模型（BRIA 商业级）
# ============================================================================

def _get_rmbg_model():
    """获取 RMBG 模型（缓存）"""
    global _RMBG_MODEL, _RMBG_PROCESSOR, _RMBG_DEVICE

    if _RMBG_MODEL is None:
        from transformers import AutoModelForImageSegmentation
        from torchvision import transforms
        import torch

        # 检测设备
        _RMBG_DEVICE = _get_device()

        print("[MODEL] Loading RMBG-1.4 model...")
        _RMBG_MODEL = AutoModelForImageSegmentation.from_pretrained(
            "briaai/RMBG-1.4",
            trust_remote_code=True
        )
        _RMBG_MODEL.to(_RMBG_DEVICE)  # 移动到 GPU/CPU
        _RMBG_MODEL.eval()
        print(f"[MODEL] Model loaded on {_RMBG_DEVICE}")

        # RMBG 使用自定义的预处理
        _RMBG_PROCESSOR = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
        ])

    return _RMBG_MODEL, _RMBG_PROCESSOR, _RMBG_DEVICE


def _rmbg_remove_background(
    image_path: Path,
    max_edge: int = 1024
) -> tuple[Image.Image, np.ndarray]:
    """
    使用 RMBG-1.4 移除背景

    Args:
        image_path: 图片路径
        max_edge: 最大边长，用于缩放加速

    Returns:
        rgba: 带 alpha 通道的 RGBA 图像
        mask: mask 数组 (0-255)
    """
    import torch
    import time

    start_time = time.time()

    model, processor, device = _get_rmbg_model()

    # 读取图片
    pil_image = Image.open(image_path).convert("RGB")
    orig_width, orig_height = pil_image.size

    # 预处理
    input_tensor = processor(pil_image).unsqueeze(0).to(device)  # 移动到 GPU

    # 推理
    with torch.no_grad():
        output = model(input_tensor)

    # 获取 mask (移回 CPU)
    mask_tensor = output[0].squeeze().cpu().numpy()
    mask = (mask_tensor * 255).astype(np.uint8)

    elapsed = time.time() - start_time
    print(f"[TIME] RMBG inference time: {elapsed:.2f}s on {device}")

    # 将 mask 缩放回原始尺寸
    mask_image = Image.fromarray(mask)
    mask_image = mask_image.resize((orig_width, orig_height), Image.Resampling.BILINEAR)
    mask = np.array(mask_image)

    # 生成 RGBA 图像
    rgba = pil_image.convert("RGBA")
    rgba.putalpha(Image.fromarray(mask))

    return rgba, mask


# ============================================================================
# GrabCut 算法（作为备选）
# ============================================================================

def _load_image(image_path: Path, max_edge: int = 1600) -> tuple[np.ndarray, Image.Image, float]:
    """加载并缩放图片"""
    pil_image = Image.open(image_path).convert("RGB")
    width, height = pil_image.size
    scale = min(1.0, max_edge / max(width, height))
    if scale < 1.0:
        pil_image = pil_image.resize(
            (int(width * scale), int(height * scale)),
            Image.Resampling.LANCZOS
        )
    bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    return bgr, pil_image, scale


def _detect_foreground_rect(bgr: np.ndarray) -> tuple[int, int, int, int]:
    """自适应前景矩形检测"""
    height, width = bgr.shape[:2]
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 30, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        margin_x = int(width * 0.08)
        margin_y = int(height * 0.08)
        return (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    shrink = 0.05
    x = int(x + w * shrink)
    y = int(y + h * shrink)
    w = int(w * (1 - 2 * shrink))
    h = int(h * (1 - 2 * shrink))
    return (max(1, x), max(1, y), max(1, w), max(1, h))


def _feather_mask(mask: np.ndarray, feather_width: int = 15) -> np.ndarray:
    """羽化边缘"""
    blurred = cv2.GaussianBlur(mask, (feather_width, feather_width), 0)
    return np.clip(blurred, 0, 255).astype(np.uint8)


def _grabcut_mask(bgr: np.ndarray, iterations: int = 12) -> np.ndarray:
    """GrabCut 算法"""
    height, width = bgr.shape[:2]
    rect = _detect_foreground_rect(bgr)
    mask = np.zeros((height, width), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
    foreground = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
    ).astype("uint8")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    foreground = _feather_mask(foreground, 15)
    return foreground


# ============================================================================
# 连通域分析
# ============================================================================

def _extract_components(
    mask: np.ndarray,
    min_area_ratio: float = 0.01,
    max_candidates: int = 3,
) -> list[tuple[int, int, int, int, np.ndarray]]:
    """提取连通域组件"""
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total_area = mask.shape[0] * mask.shape[1]
    min_area = int(total_area * min_area_ratio)
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
    return components[:max_candidates]


# ============================================================================
# 辅助函数
# ============================================================================

def _classify_source_type(width: int, height: int) -> str:
    """根据宽高比分类主体类型"""
    ratio = width / max(height, 1)
    if ratio < 0.85:
        return "cup"
    if ratio > 1.35:
        return "plate"
    return "bowl"


def _save_candidate_files(
    rgb_image: Image.Image,
    component_mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    output_dir: Path,
    item_id: str,
) -> tuple[Path, Path, Path]:
    """保存候选结果文件"""
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


# ============================================================================
# 主入口
# ============================================================================

def generate_cutout_candidates(
    image_path: Path,
    output_dir: Path,
    task_id: str,
    algorithm: AlgorithmType | None = None,
) -> list[CutoutCandidate]:
    """
    生成抠图候选结果

    Args:
        image_path: 输入图片路径
        output_dir: 输出目录
        task_id: 任务 ID
        algorithm: 算法选择 ('rmbg', 'birefnet', 'isnet', 'u2net', 'grabcut')

    Returns:
        候选结果列表
    """
    config = CUTOUT_CONFIG.copy()
    if algorithm is None:
        algorithm = config.get('algorithm', 'rmbg')

    output_dir.mkdir(parents=True, exist_ok=True)

    # 根据算法选择处理方式
    if algorithm == 'rmbg':
        # RMBG-1.4 商业级模型
        rgba, mask = _rmbg_remove_background(image_path, config['max_edge'])
        pil_image = rgba.convert('RGB')
    elif algorithm in ['birefnet', 'isnet', 'u2net']:
        # Rembg 系模型
        model_name = config['rembg_models'].get(algorithm, 'birefnet-general')
        rgba, mask = _rembg_remove_background(
            image_path, model_name, max_edge=config.get('max_edge', 768)
        )
        pil_image = rgba.convert('RGB')
    else:
        # GrabCut
        bgr, pil_image, scale = _load_image(image_path, config['max_edge'])
        mask = _grabcut_mask(bgr, iterations=config['grabcut_iterations'])

    # 提取连通域
    components = _extract_components(
        mask,
        min_area_ratio=config['min_component_area_ratio'],
        max_candidates=config['max_candidates'],
    )

    # 生成候选结果
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
        score = min(0.99, 0.70 + area / max(image_area, 1))

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
                algorithm=algorithm,
            )
        )

    return candidates


def candidate_to_dict(candidate: CutoutCandidate, base_url: str) -> dict[str, Any]:
    """转换为字典格式"""
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
        "algorithm": candidate.algorithm,
    }
