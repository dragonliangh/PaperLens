# download_model.py
from modelscope import snapshot_download

# 下载 tokenizer 相关文件到本地 models/bge-small-en-v1.5 文件夹
model_dir = snapshot_download(
    'BAAI/bge-small-en-v1.5',
    local_dir='./models/bge-small-en-v1.5'
)
print(f"文件已下载到: {model_dir}")