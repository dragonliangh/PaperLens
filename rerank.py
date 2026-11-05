"""PaperLens 第 6 块：Cross-Encoder 精排。

【模块职责】
  对第 2 块向量检索召回的候选段落（top-20）进行精细打分，剔除语义相关但
  不直接回答问题的噪声，选出最可能包含答案的段落（top-4）。
  内部流程：
    (query, candidates) -> 拼接 (query, doc) 对 -> Cross-Encoder 打分
    -> 按分数降序 -> 截取 final_k -> 返回重排后的结果

【在架构中的位置】
  第 6 块，位于检索层（第 2 块）和生成层（第 3 块）之间。
    - 上游依赖：index_store.RetrievedChunk（粗排结果）。
    - 下游被依赖：qa.answer() 调用本模块，将重排后的段落拼进 LLM Prompt。
    - 第 7 块评估脚本会单独调用本模块，衡量重排对 Recall / MRR 的提升。

【对外暴露】
  - rerank(query, candidates, final_k=4) -> list[RetrievedChunk]

【依赖】
  - sentence-transformers ：提供 CrossEncoder 类。
  - torch ：用于强制类型转换和分数归一化（Sigmoid）。
"""
import os
# 【关键补丁】PyTorch 2.6+ 默认 weights_only=True，会拒绝加载 .bin 权重中的非张量对象。
# 设置此环境变量强制关闭 weights_only，必须放在 import torch / transformers 之前。
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# =============================================================================
# 【底层强拦截】统一全局环境，确保 CrossEncoder 加载 .bin 时不被拦截。
# =============================================================================
import torch
import torch.serialization

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

_original_serialization_load = torch.serialization.load
def _patched_serialization_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return _original_serialization_load(*args, **kwargs)
torch.serialization.load = _patched_serialization_load
# =============================================================================

from functools import lru_cache
from sentence_transformers import CrossEncoder

from index_store import RetrievedChunk

# 模型本地路径（需提前下载到 models/bge-reranker-base/）。
MODEL_PATH = "./models/bge-reranker-base"


@lru_cache(maxsize=1)
def _get_model() -> CrossEncoder:
    """加载并缓存 Cross-Encoder 模型，进程内只加载一次。

    为什么缓存：
      模型文件约 1.1GB，加载需要数秒到十几秒。
      重排函数在每次提问时都会调用，用 lru_cache 避免重复加载的开销。

    性能隐患 / 假设：
      假设 MODEL_PATH 下的文件完整（config.json, model.safetensors 等）。
      若文件缺失，这里会抛异常，按契约不吞异常。

    【云端修复】强制使用 pytorch_model.bin：
      云端 Git LFS 拉取的 model.safetensors 可能存在文件头损坏
      (Error while deserializing header: header too large)，
      通过 model_kwargs 禁用 safetensors，强制回退到 .bin 权重。
    """
    return CrossEncoder(MODEL_PATH, model_kwargs={"use_safetensors": False})


def rerank(query: str, candidates: list[RetrievedChunk], final_k: int = 4) -> list[RetrievedChunk]:
    """对候选段落进行精排，返回最相关的 final_k 个。

    参数：
      query      : 用户原始问题。
      candidates : index_store.search() 返回的粗排列表（通常 top_k=20）。
      final_k    : 最终保留的段落数，默认 4。

    返回：
      重排后的 RetrievedChunk 列表，长度 ≤ final_k。
      【核心逻辑】返回的 score 字段已被替换为 Cross-Encoder 的 0~1 相关性分数，
      不再是余弦相似度。第 7 块设拒答阈值时必须使用此分数。

    错误语义：
      - 模型文件缺失：_get_model() 抛异常，本函数不捕获。
      - 空 query 或空候选：返回 []。
    """
    # 边界处理：空 query 或空候选，直接返回空列表，避免无意义地调模型。
    if not query.strip() or not candidates:
        return []

    model = _get_model()

    # 【核心逻辑】Cross-Encoder 需要 (query, document) 对作为输入。
    # 拼接格式为 [query, text] 列表的列表。
    pairs = [[query, c.chunk.text] for c in candidates]

    # 【性能隐患】predict 会逐对进行前向传播。20 个 pair 在 CPU 上约 0.5~1 秒。
    # batch_size 默认 32，若显存/内存不足可手动调小。
    with torch.no_grad():
        # 【性能修复】强制设置 batch_size=4。
        # 默认通常是 32，对于 400 token 的论文段落，CPU 内存扛不住 32 个 pair 的并发矩阵，
        # 会导致极度的内存交换（Swap），实测不加此参数会从 1 秒飙升到 25 秒。
        raw_scores = model.predict(pairs, batch_size=4, show_progress_bar=False)

    # 【核心逻辑】将 raw logits 经 sigmoid 映射到 0~1，方便第 7 块设阈值。
    # 由于 sentence-transformers 不同版本可能返回 list、ndarray 或 Tensor，
    # 这里先统一强制转为 torch.Tensor，消除 Pylance 类型推导歧义，并避免 numpy 报错。
    if not isinstance(raw_scores, torch.Tensor):
        raw_scores = torch.tensor(raw_scores)

    # 如果模型版本已经自带激活（输出在 0~1 之间），则跳过 sigmoid。
    # 这里的 .max() 和 .min() 返回的是 Tensor 标量，与浮点数比较是安全的。
    if raw_scores.max() > 1.0 or raw_scores.min() < 0.0:
        norm_scores = torch.sigmoid(raw_scores)
    else:
        norm_scores = raw_scores

    # 将新分数合并回 RetrievedChunk，并替换旧的 score 字段。
    # 【假设特定输入格式】candidates 与 norm_scores 长度一致，顺序对应。
    scored = []
    for c, score in zip(candidates, norm_scores):
        # 直接新建对象，保持显式；确保 score 转为 Python float。
        c_reranked = RetrievedChunk(
            chunk=c.chunk,
            score=float(score),
        )
        scored.append(c_reranked)

    # 按新分数降序排序，并截取 final_k。
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored[:final_k]