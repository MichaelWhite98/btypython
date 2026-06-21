# 云端抠图 API 集成方案

## 背景

当前本地 BiRefNet/U2-Net 模型在 CPU 环境下推理速度较慢（5-15 秒/张），影响用户体验。云端 API 可提供：

- ⚡ 更快的响应速度（1-3 秒）
- ☁️ 无需本地 GPU
- 🔄 稳定的服务质量
- 📈 按需扩展

---

## 云服务商对比

| 服务商 | API 名称 | 响应时间 | 价格 | 免费额度 | 推荐度 |
|--------|---------|---------|------|---------|--------|
| **阿里云** | 人像/商品分割 | 1-2s | ¥0.01-0.05/次 | 1000次/月 | ⭐⭐⭐⭐⭐ |
| **腾讯云** | 图像分割 | 1-2s | ¥0.01-0.05/次 | 1000次/月 | ⭐⭐⭐⭐⭐ |
| **百度智能云** | 主体检测 | 1-3s | ¥0.008/次 | 500次/天 | ⭐⭐⭐⭐ |
| **火山引擎** | 火山抠图 | 1-2s | ¥0.02/次 | 100次/月 | ⭐⭐⭐ |

### 推荐选择

**首推：阿里云或腾讯云**
- 免费额度足够开发测试
- 文档完善，SDK 成熟
- 响应速度快
- 支持多种分割类型（人像、商品、通用）

---

## 架构设计

```
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   小程序     │───►│  btypython      │───►│  云端 API       │
│  BTYMini    │◄───│  (代理层)        │◄───│  阿里云/腾讯云   │
└─────────────┘    └─────────────────┘    └─────────────────┘
                          │
                          ▼
                   ┌─────────────────┐
                   │  本地模型备选    │
                   │  (降级方案)      │
                   └─────────────────┘
```

### 优势

1. **小程序端无需改动** - API 接口保持不变
2. **自动降级** - 云端失败时切换本地模型
3. **成本可控** - 优先云端，本地备用

---

## API 接口设计

### 请求参数扩展

```python
# POST /api/cutout/tasks
{
    "scene": "food-diary-cutout",
    "imageUrl": "https://...",
    "algorithm": "cloud-aliyun",  # 新增云端选项
    "fallback": true              # 是否启用本地降级
}
```

### 支持的 algorithm 值

| 值 | 说明 |
|---|------|
| `birefnet` | 本地 BiRefNet（当前默认） |
| `u2net` | 本地 U2-Net（较快） |
| `grabcut` | 本地 GrabCut（最快，效果一般） |
| `cloud-aliyun` | 阿里云分割 API |
| `cloud-tencent` | 腾讯云分割 API |
| `cloud-baidu` | 百度主体检测 API |
| `cloud-auto` | 自动选择（优先云端，失败降级） |

---

## 阿里云集成方案（推荐实现）

### 1. 开通服务

```
阿里云控制台 → 视觉智能开放平台 → 图像分割
https://vision.aliyun.com/segment
```

### 2. 获取 AccessKey

```
阿里云控制台 → AccessKey 管理 → 创建 AccessKey
```

### 3. 安装 SDK

```bash
pip install aliyun-python-sdk-core aliyun-python-sdk-viapiutils
# 或使用新版 SDK
pip install alibabacloud-viapi20230117
```

### 4. 配置文件

```python
# config/cloud_providers.py
ALIYUN_CONFIG = {
    "access_key_id": "your-access-key-id",
    "access_key_secret": "your-access-key-secret",
    "endpoint": "viapi.cn-shanghai.aliyuncs.com",
    "region": "cn-shanghai",
}
```

### 5. 核心代码

```python
# app/cloud/aliyun_segment.py
from alibabacloud_viapi20230117.client import Client
from alibabacloud_tea_openapi import models as open_api_models
from alibabacloud_viapi20230117 import models as viapi_models
from alibabacloud_tea_util import models as util_models
import base64
from pathlib import Path
from PIL import Image
from io import BytesIO
import requests


class AliyunSegmentService:
    """阿里云图像分割服务"""

    def __init__(self, config: dict):
        self.config = config
        self.client = self._create_client()

    def _create_client(self) -> Client:
        config = open_api_models.Config(
            access_key_id=self.config["access_key_id"],
            access_key_secret=self.config["access_key_secret"],
        )
        config.endpoint = self.config["endpoint"]
        return Client(config)

    def segment(self, image_url: str, segment_type: str = "general") -> bytes:
        """
        调用分割 API

        Args:
            image_url: 图片 URL 或本地路径
            segment_type: 分割类型
                - "general": 通用分割
                - "portrait": 人像分割
                - "commodity": 商品分割

        Returns:
            PNG 图片 bytes（带透明通道）
        """
        # 构建 OSS URL（阿里云要求图片为可访问的 URL）
        # 如果是本地文件，需要先上传到 OSS
        if image_url.startswith("http"):
            url = image_url
        else:
            url = self._upload_to_oss(image_url)

        # 选择 API
        if segment_type == "portrait":
            request = viapi_models.SegmentPersonRequest(image_url=url)
            response = self.client.segment_person(request)
        elif segment_type == "commodity":
            request = viapi_models.SegmentCommodityRequest(image_url=url)
            response = self.client.segment_commodity(request)
        else:
            # 通用分割
            request = viapi_models.SegmentCommonImageRequest(image_url=url)
            response = self.client.segment_common_image(request)

        # 返回结果 URL，下载图片
        result_url = response.body.data.image_url
        return self._download_image(result_url)

    def _upload_to_oss(self, local_path: str) -> str:
        """上传本地图片到 OSS，返回 URL"""
        # TODO: 实现 OSS 上传
        pass

    def _download_image(self, url: str) -> bytes:
        """下载图片"""
        response = requests.get(url, timeout=30)
        return response.content
```

### 6. 集成到现有服务

```python
# app/cutout.py 新增函数

def cloud_segment(
    image_url: str,
    provider: str = "aliyun",
    segment_type: str = "commodity"
) -> tuple[Image.Image, np.ndarray]:
    """
    云端抠图

    Args:
        image_url: 图片 URL
        provider: 云服务商 (aliyun/tencent/baidu)
        segment_type: 分割类型

    Returns:
        rgba: RGBA 图像
        mask: mask 数组
    """
    from .cloud.aliyun_segment import AliyunSegmentService
    from .cloud.tencent_segment import TencentSegmentService
    from .cloud.baidu_segment import BaiduSegmentService

    providers = {
        "aliyun": AliyunSegmentService,
        "tencent": TencentSegmentService,
        "baidu": BaiduSegmentService,
    }

    service = providers[provider](get_cloud_config(provider))
    image_bytes = service.segment(image_url, segment_type)

    # 转换为 PIL Image
    rgba = Image.open(BytesIO(image_bytes)).convert("RGBA")
    alpha = rgba.split()[-1]
    mask = np.array(alpha)

    return rgba, mask
```

---

## 腾讯云集成方案

### 1. 开通服务

```
腾讯云控制台 → 图像分析 → 图像分割
https://console.cloud.tencent.com/ai/image/segment
```

### 2. 获取密钥

```
腾讯云控制台 → 访问管理 → API密钥管理
```

### 3. 安装 SDK

```bash
pip install tencentcloud-sdk-python
```

### 4. 核心代码

```python
# app/cloud/tencent_segment.py
from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tiia.v20190529 import tiia_client, models
import base64
from pathlib import Path


class TencentSegmentService:
    """腾讯云图像分割服务"""

    def __init__(self, config: dict):
        self.config = config
        self.client = self._create_client()

    def _create_client(self):
        cred = credential.Credential(
            self.config["secret_id"],
            self.config["secret_key"]
        )
        http_profile = HttpProfile(endpoint="tiia.tencentcloudapi.com")
        client_profile = ClientProfile(httpProfile=http_profile)
        return tiia_client.TiiaClient(cred, "ap-guangzhou", client_profile)

    def segment(self, image_url: str, segment_type: str = "general") -> bytes:
        """
        调用分割 API

        Args:
            image_url: 图片 URL 或 base64
            segment_type: 分割类型

        Returns:
            PNG 图片 bytes
        """
        request = models.SegmentPicRequest()

        # 如果是 URL
        if image_url.startswith("http"):
            request.ImageUrl = image_url
        else:
            # 本地文件转 base64
            with open(image_url, "rb") as f:
                request.ImageBase64 = base64.b64encode(f.read()).decode()

        # 设置分割类型
        # 0: 人像  1: 商品  2: 通用
        request.SegmentType = {"portrait": 0, "commodity": 1, "general": 2}.get(segment_type, 2)

        response = self.client.SegmentPic(request)

        # 返回 base64 解码后的图片
        return base64.b64decode(response.ResultImage)
```

---

## 成本估算

### 开发测试阶段

- 免费额度足够：阿里云 1000 次/月 + 腾讯云 1000 次/月
- 预估用量：< 500 次/月

### 生产环境

假设日活 100 用户，每用户 3 张图片：

| 指标 | 数值 |
|------|------|
| 日调用量 | 300 次 |
| 月调用量 | 9,000 次 |
| 单价 | ¥0.02/次 |
| 月成本 | ¥180 |
| 年成本 | ¥2,160 |

**优化建议**：
- 开启本地缓存，相同图片不重复调用
- 图片压缩后再上传，减少流量费用
- 高峰期用云端，低峰期用本地

---

## 实施计划

### 阶段一：环境准备（1 天）

- [ ] 注册阿里云/腾讯云账号
- [ ] 开通图像分割服务
- [ ] 获取 API 密钥
- [ ] 安装 SDK

### 阶段二：代码开发（2-3 天）

- [ ] 创建 `app/cloud/` 目录结构
- [ ] 实现阿里云分割服务
- [ ] 实现腾讯云分割服务（备选）
- [ ] 修改 `cutout.py` 支持云端调用
- [ ] 实现自动降级逻辑

### 阶段三：测试验证（1 天）

- [ ] 单元测试
- [ ] 接口测试
- [ ] 小程序端联调
- [ ] 性能对比测试

### 阶段四：上线部署（1 天）

- [ ] 配置生产环境密钥
- [ ] 更新文档
- [ ] 监控告警配置

---

## 目录结构

```
btypython/
├── app/
│   ├── cloud/                      # 新增：云端服务
│   │   ├── __init__.py
│   │   ├── base.py                 # 基类定义
│   │   ├── aliyun_segment.py       # 阿里云分割
│   │   ├── tencent_segment.py      # 腾讯云分割
│   │   └── baidu_segment.py        # 百度分割（可选）
│   ├── config/                     # 新增：配置管理
│   │   ├── __init__.py
│   │   └── cloud_providers.py
│   ├── cutout.py                   # 修改：支持云端
│   └── main.py                     # 修改：新增配置接口
├── docs/
│   ├── cutout-optimization-plan.md
│   └── cloud-integration-plan.md   # 本文档
└── requirements.txt                # 更新依赖
```

---

## 下一步

1. **确认云服务商**：选择阿里云或腾讯云（推荐阿里云）
2. **提供 API 密钥**：我帮你完成代码集成
3. **测试验证**：对比云端与本地效果和速度
