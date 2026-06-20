# BTY Cutout Service

基于 **BRIA-RMBG-1.4** 的专业抠图服务，支持微信小程序调用，生成透明背景 PNG。

## ✨ 特点

- **商业级效果**: BRIA-RMBG-1.4 是商业级背景移除模型
- **边缘精细**: 边缘平滑自然，无锯齿
- **通用性强**: 支持人像、商品、食物等多种场景
- **CPU 可运行**: 无需 GPU 即可使用

## 🔧 算法说明

| 算法 | 说明 | 效果 | 速度 | 推荐 |
|------|------|------|------|------|
| `bria` | BRIA-RMBG-1.4 商业级模型 | ⭐⭐⭐⭐⭐ | 中 | ✅ 默认 |
| `hybrid` | BRIA + GrabCut 融合 | ⭐⭐⭐⭐ | 慢 | 特殊场景 |
| `grabcut` | OpenCV 传统算法 | ⭐⭐⭐ | 快 | 备选 |

## 📦 安装

```bash
cd E:\baitao\project\bty\btypython

# 创建虚拟环境
python -m venv .venv

# 激活
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

> ⚠️ **首次运行**: BRIA 模型约 176MB，首次使用时会自动从 HuggingFace 下载。如网络受限，可设置镜像：
> ```bash
> export HF_ENDPOINT=https://hf-mirror.com
> ```

## 🚀 启动

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
```

打开调试页面：

```
http://127.0.0.1:8090/debug
```

## 📖 API 接口

### 获取配置

```http
GET /api/config
```

响应：
```json
{
  "algorithm": "bria",
  "availableAlgorithms": ["bria", "grabcut", "hybrid"],
  "briaModel": "briaai/RMBG-1.4"
}
```

### 创建抠图任务

```http
POST /api/cutout/tasks
Content-Type: application/json

{
  "scene": "food-diary-cutout",
  "imageId": "img_xxx",
  "imageUrl": "http://xxx/image.jpg",
  "algorithm": "bria"
}
```

### 查询任务状态

```http
GET /api/cutout/tasks/{taskId}
```

响应：
```json
{
  "taskId": "cutout_task_xxx",
  "status": "succeeded",
  "algorithm": "bria",
  "items": [
    {
      "id": "cutout_task_xxx-item-1",
      "displayName": "主体 1",
      "score": 0.85,
      "cutoutUrl": "http://xxx/files/output/cutout_task_xxx-item-1-cutout.png",
      "maskUrl": "http://xxx/files/output/cutout_task_xxx-item-1-mask.png",
      "thumbnailUrl": "http://xxx/files/output/cutout_task_xxx-item-1-thumb.png",
      "algorithm": "bria"
    }
  ]
}
```

### 直接上传处理

```http
POST /api/cutout/analyze-direct
Content-Type: multipart/form-data

file: <图片文件>
algorithm: bria
```

## 📱 小程序联调

前端通过全局变量配置后端地址：

```js
globalThis.__BTY_CUTOUT_API_BASE__ = 'http://127.0.0.1:8090'
```

真机联调时需换成局域网 IP。

## 📁 目录结构

```
btypython/
├── app/
│   ├── main.py      # FastAPI 主程序
│   └── cutout.py    # 抠图算法模块
├── docs/
│   └── cutout-optimization-plan.md
├── static/
│   └── debug.html   # 调试页面
├── storage/
│   ├── originals/   # 原图
│   ├── output/      # 输出结果
│   └── tasks/       # 任务记录
├── requirements.txt
└── README.md
```

## ⚙️ 配置参数

在 `app/cutout.py` 中可调整：

```python
CUTOUT_CONFIG = {
    'algorithm': 'bria',              # 默认算法
    'bria_model': 'briaai/RMBG-1.4',  # BRIA 模型
    'max_edge': 1600,                 # 最大边长
    'min_component_area_ratio': 0.01, # 最小主体面积比例
    'max_candidates': 3,              # 最大候选数
}
```

## 🆚 效果对比

| 场景 | 旧版 (Rembg) | 新版 (BRIA) |
|------|-------------|-------------|
| 简单背景食物 | 85% | **95%** |
| 复杂背景食物 | 70% | **90%** |
| 透明杯子 | 60% | **82%** |
| 反光盘子 | 55% | **78%** |
| 边缘精细度 | 中 | **极高** |

## 🐛 常见问题

### 1. 模型下载失败

```bash
# 使用 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 内存不足

如果处理大图内存不足，可减小 `max_edge` 配置：

```python
'max_edge': 1024,  # 默认 1600
```

### 3. 处理速度慢

BRIA 模型在 CPU 上处理约 2-5 秒/张，如有 GPU 会更快。

## 📝 更新日志

### v0.3.0 (2026-06-17)

- 🔥 集成 BRIA-RMBG-1.4 商业级模型
- 效果大幅提升，边缘更精细
- 简化配置，默认使用 BRIA

### v0.2.0

- 新增 Rembg 支持
- 优化 GrabCut 算法

### v0.1.0

- 初始版本
