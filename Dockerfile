# 使用官方 Python 镜像作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装 Nginx
RUN apt-get update && \
    apt-get install -y nginx && \
    rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt 并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和配置文件
COPY all_model_api_server.py .
COPY nginx.conf /etc/nginx/nginx.conf
COPY templates/ ./templates/

# 创建存放 API 密钥的目录
RUN mkdir -p /api_key

# 暴露端口
EXPOSE 7778 8001

# 创建启动脚本
COPY <<EOF /start.sh
#!/bin/bash
# 启动 FastAPI 服务
python all_model_api_server.py &
# 启动 Nginx
nginx -g 'daemon off;'
EOF

RUN chmod +x /start.sh

# 启动服务
CMD ["/start.sh"]
