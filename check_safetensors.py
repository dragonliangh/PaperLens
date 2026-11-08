from safetensors import safe_open

def check(path):
    try:
        with safe_open(path, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            print(f"✅ {path} 完好，包含 {len(keys)} 个张量")
    except Exception as e:
        print(f"❌ {path} 损坏：{e}")

check("./models/bge-small-en-v1.5/model.safetensors")
check("./models/bge-reranker-base/model.safetensors")