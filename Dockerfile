# 基础镜像：slim 版本体积小
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统底层依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 PyTorch (CPU版) 和 Modelscope
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple

# 【核心修改】在构建镜像时自动下载模型，保证云端有完整的 .bin 文件
RUN python -c "from modelscope import snapshot_download; \
    snapshot_download('BAAI/bge-small-en-v1.5', local_dir='./models/bge-small-en-v1.5'); \
    snapshot_download('BAAI/bge-reranker-base', local_dir='./models/bge-reranker-base')"

COPY requirements.txt .
# 安装剩余依赖
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]