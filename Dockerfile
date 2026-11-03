# 基础镜像：slim 版本体积小，如果后续遇到 C 库缺失，可改为 python:3.13
FROM python:3.13-slim

# 设置工作目录
WORKDIR /app

# 安装必要的系统底层依赖：
# - build-essential: 防止某些 Python 包需要编译源码
# - libgomp1: PyTorch (通过 torch) 运行必须依赖的 OpenMP 库
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝 requirements.txt 并安装依赖（利用 Docker 缓存层加速）
COPY requirements.txt .
# 使用清华源加速，避免国内构建时网络超时
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 拷贝整个项目代码（.dockerignore 会自动排除不需要的文件）
COPY . .

# 暴露端口（魔搭强制要求 7860）
EXPOSE 7860

# 启动命令
CMD ["python", "app.py"]