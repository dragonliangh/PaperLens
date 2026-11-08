# 基础镜像：slim 版本体积小
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统底层依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 拷贝 requirements.txt
COPY requirements.txt .

# 【核心逻辑】单独从 PyTorch 官方 CPU 源安装，剥离巨大的 CUDA 依赖
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 安装剩余依赖（包含 modelscope，用于下一步下载模型）
RUN pip install --no-cache-dir modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 【核心修复】在构建镜像时，通过魔搭内网极速下载完整的 .bin 模型文件
# 这样即使 Git 仓库里没有 models/，云端环境里也绝对有一份完好无损的模型
RUN python -c "from modelscope import snapshot_download; \
    snapshot_download('BAAI/bge-small-en-v1.5', local_dir='./models/bge-small-en-v1.5'); \
    snapshot_download('BAAI/bge-reranker-base', local_dir='./models/bge-reranker-base')"

# 拷贝项目代码（此时 .gitignore 会自动排除本地庞大的 models/ 文件夹）
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]