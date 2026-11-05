"""
本地最终验证脚本。
【目的】
  1. 把 .bin 权重重新转成干净的 .safetensors，覆盖云端损坏的文件。
  2. 验证分块、建索引、检索、重排全链路。
【原理】
  safetensors 是纯张量格式，加载时不会调用 torch.load，
  从根本上绕开 PyTorch 2.6+ 的 weights_only 限制。
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from safetensors.torch import save_file


def regen_safetensors(model_dir: str):
    """从 .bin 重新生成 .safetensors，覆盖损坏的文件。
    
    关键点：torch.load 显式传 weights_only=False，
    强制允许加载包含自定义对象的 pickle 文件。
    """
    bin_path = os.path.join(model_dir, "pytorch_model.bin")
    safe_path = os.path.join(model_dir, "model.safetensors")

    if not os.path.exists(bin_path):
        print(f"  跳过（没有 .bin 文件）：{model_dir}")
        return

    print(f"[1/2] 加载 {bin_path} ...")
    state_dict = torch.load(bin_path, map_location="cpu", weights_only=False)
    print(f"  OK，共 {len(state_dict)} 个键")

    # safetensors 要求所有值都是连续的 torch.Tensor
    clean = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v.contiguous()

    print(f"[2/2] 保存到 {safe_path} ...")
    save_file(clean, safe_path)
    print(f"  完成，共保存 {len(clean)} 个张量")


print("=" * 50)
print("Step 1: 重新生成 safetensors")
print("=" * 50)
regen_safetensors("./models/bge-small-en-v1.5")
regen_safetensors("./models/bge-reranker-base")

print("\n" + "=" * 50)
print("Step 2: 端到端验证")
print("=" * 50)

from ingest import parse_and_chunk, _get_tokenizer
from index_store import build_index, search, clear_index, _get_model
from rerank import rerank, _get_model as _get_reranker

print("加载 tokenizer...")
tok = _get_tokenizer()
print(f"  OK，词表大小 {len(tok)}")

print("加载嵌入模型...")
m = _get_model()
print(f"  OK，维度 {m.get_sentence_embedding_dimension()}")

print("加载重排模型...")
rm = _get_reranker()
print("  OK")

print("\n构建索引...")
clear_index()
chunks = parse_and_chunk('data/sample.pdf')
print(f"  分块：{len(chunks)} 个 chunk")
build_index(chunks)
print("  索引构建完成")

q = 'How much does LoRA reduce trainable parameters?'
print(f"\n检索：{q}")
res = search(q, top_k=20)
print("--- 重排前 top-5 ---")
for r in res[:5]:
    print(f'  p{r.chunk.page_start} | score={r.score:.4f}')

reranked = rerank(q, res, final_k=4)
print("--- 重排后 top-4 ---")
for r in reranked:
    print(f'  p{r.chunk.page_start} | score={r.score:.4f}')

# 断言：正确答案的 p1 或 p5 必须出现在 top-4
top_pages = {r.chunk.page_start for r in reranked}
assert top_pages & {1, 5}, f"重排失败！含答案的 p1/p5 不在 top-4 中，实际是 {top_pages}"

print("\n" + "=" * 50)
print("✅ 全部测试通过！可以推送了。")
print("=" * 50)