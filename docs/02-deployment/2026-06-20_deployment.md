# BTY Cutout Service - 生产环境部署

## 部署方式

### 方式一：直接部署

```bash
# 1. 创建虚拟环境
python3 -m venv /opt/bty-cutout/venv
source /opt/bty-cutout/venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 3. 预下载模型（避免首次请求慢）
python -c "from rembg import new_session; new_session('birefnet-general')"

# 4. 使用 Gunicorn 启动（生产级 WSGI 服务器）
pip install gunicorn
gunicorn app.main:app \
  -w 2 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8090 \
  --timeout 120 \
  --keep-alive 5 \
  --access-logfile /var/log/bty-cutout/access.log \
  --error-logfile /var/log/bty-cutout/error.log
```

### 方式二：Docker 部署（推荐）

```bash
# 构建镜像
docker build -t bty-cutout:latest .

# 运行容器
docker run -d \
  --name bty-cutout \
  -p 8090:8090 \
  -v /data/bty-cutout/storage:/app/storage \
  -v /data/bty-cutout/models:/root/.u2net \
  --restart unless-stopped \
  bty-cutout:latest
```

### 方式三：Systemd 服务

```bash
# 创建服务文件
sudo nano /etc/systemd/system/bty-cutout.service
```

```ini
[Unit]
Description=BTY Cutout Service
After=network.target

[Service]
Type=simple
User=www-data
Group=www-data
WorkingDirectory=/opt/bty-cutout
Environment="PATH=/opt/bty-cutout/venv/bin"
ExecStart=/opt/bty-cutout/venv/bin/gunicorn app.main:app \
  -w 2 \
  -k uvicorn.workers.UvicornWorker \
  -b 0.0.0.0:8090 \
  --timeout 120
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable bty-cutout
sudo systemctl start bty-cutout
```

## 性能优化

### 1. 使用 GPU 加速

```bash
# 安装 CUDA 版本的 onnxruntime
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

### 2. 调整 Worker 数量

```bash
# CPU 核心数 - 1，最多 4 个
gunicorn app.main:app -w 4 ...
```

### 3. 添加 Nginx 反向代理

```nginx
upstream bty_cutout {
    server 127.0.0.1:8090;
    keepalive 32;
}

server {
    listen 80;
    server_name cutout.yourdomain.com;

    client_max_body_size 20M;

    location / {
        proxy_pass http://bty_cutout;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 60s;
        proxy_read_timeout 120s;
    }
}
```

## 注意事项

1. **模型预下载**: 首次启动前预下载模型，避免首次请求超时
2. **内存配置**: BiRefNet 模型约需 2GB 内存，建议服务器 4GB+
3. **超时设置**: 图片处理可能需要 5-30 秒，设置足够超时时间
4. **存储清理**: 定期清理 storage 目录下的旧文件
5. **日志轮转**: 配置日志轮转避免磁盘占满

## 监控

```bash
# 健康检查
curl http://localhost:8090/health

# 查看配置
curl http://localhost:8090/api/config
```
