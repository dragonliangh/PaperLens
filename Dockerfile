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

# 单独从 PyTorch 官方 CPU 源安装
RUN pip install --no-cache-dir torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu

# 安装剩余依赖（双源策略：清华源优先；该源在构建容器内不可达时回退官方 PyPI）
# 背景：魔搭构建曾出现 0.71 秒返回 from versions: none —— 是索引没返回任何候选，不是版本缺失
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple --retries 5 --timeout 60 \
 || pip install --no-cache-dir -r requirements.txt --retries 5 --timeout 60

# 注意：这里不再下载模型，镜像会变得非常小
COPY . .

EXPOSE 7860

CMD ["python", "app.py"]