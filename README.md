# BTY Cutout Service

这是一个本地可调试的 Python 主体抠图服务，目标是支持微信小程序调用，并生成透明背景 PNG。

## 当前能力

- `POST /api/media/upload-token`
- `POST /api/uploads/local`
- `POST /api/cutout/tasks`
- `GET /api/cutout/tasks/{taskId}`
- `GET /debug`

当前版本使用 `OpenCV GrabCut` 做第一版主体提取，重点是先让你本地看效果、调 API、调前端联调，不是最终模型版质量。

## 目录

```text
bty-cutout-service/
├── app/
│   ├── main.py
│   └── cutout.py
├── static/
│   └── debug.html
├── storage/
│   ├── originals/
│   ├── output/
│   └── tasks/
└── requirements.txt
```

## 启动

```bash
cd /Users/baitao/project/bty/bty-cutout-service
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8090
```

打开调试页：

```text
http://127.0.0.1:8090/debug
```

## 小程序联调

前端当前支持通过全局变量配置后端地址：

```js
globalThis.__BTY_CUTOUT_API_BASE__ = 'http://127.0.0.1:8090'
```

如果是微信开发者工具真机/模拟器联调，通常需要换成宿主机可访问地址，例如局域网 IP。

## 返回结果

任务成功后会返回多个候选主体，每个候选包含：

- `maskUrl`
- `cutoutUrl`
- `thumbnailUrl`
- `bbox`
- `areaRatio`

其中 `cutoutUrl` 是透明背景 PNG。

## 限制

- 当前版本没有真实餐具模型
- 复杂背景时可能保留过多区域
- 透明杯、反光盘子边缘效果一般
- 多个主体重叠时排序不稳定

下一步如果要做正式版本，建议把 `GrabCut` 替换成真正的检测 + 分割模型服务。
