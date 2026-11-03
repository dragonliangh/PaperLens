# 基础镜像：slim 版本体积小
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统底层依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# 【核心逻辑】单独从 PyTorch 官方 CPU 源安装，避免 Linux 下默认拉取巨大的 CUDA 依赖 (约 500MB+ 变成约 180MB)
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 安装剩余依赖（此时 torch 已满足，不会重复下载）
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]