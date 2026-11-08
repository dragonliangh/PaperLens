"""诊断剩余 8 道"粗排命中但精排丢"的题。

【背景】
  上一轮已诊断 4 道（en_016/en_029/en_042/en_044），发现：
    - en_044 是 gold 标注错（应在 p19，标成了 p18）；
    - en_016、en_029 是 gold 标注太严格（精排找到的其他页也直接回答问题）；
    - en_042 是精排没推走，只是排到了 #7。
  本脚本诊断剩余 8 道，看是否也有类似的标注问题，还是真的模型问题。

【为什么只诊断这 8 道】
  全量评估 23 分钟。这 8 道是诊断脚本里"精排排名 9 名以后"的题，
  只需要对它们跑粗排 + 精排 + 打印原文，约 2-3 分钟。

【输出结构（与 diagnose_pushed_away.py 一致）】
  对每道题：
    1. 粗排 top-5（带 gold_page 标记 ★）
    2. 精排 top-8（实际看 20，只打印 8）
    3. gold chunk 的真实排名
    4. gold chunk 与 query 拼接后的 token 长度（判断是否被截断）
    5. gold chunk 原文前 500 字符（人工看是否真的回答问题）

【遵循】
  契约 v2 第 8 条：文件/方法/语句/边界四级注释。
  只读 search / rerank / load_eval_set，不修改任何生产代码。
"""
from index_store import search
from rerank import rerank
from evaluate import load_eval_set
from ingest import _get_tokenizer

EVAL_SET_PATH = "eval_set.json"
COARSE_TOP_K = 20
# 诊断目标是上一轮诊断脚本里"精排排名 > 8"的 8 道题。
TARGET_IDS = [
    "en_003", "en_006", "en_013", "en_014",
    "en_038", "en_040", "en_045", "en_048",
]


def diagnose_remaining():
    """逐道打印粗排、精排、gold chunk 详情，供人工判断标注是否合理。"""
    items = load_eval_set(EVAL_SET_PATH)
    item_map = {it["id"]: it for it in items if it["id"] in TARGET_IDS}

    tok = _get_tokenizer()

    # 断言：确保 8 道题都在评估集里，防止 id 写错导致静默跳过
    missing = set(TARGET_IDS) - set(item_map.keys())
    if missing:
        raise ValueError(f"评估集中找不到这些 id：{missing}")

    for tid in TARGET_IDS:
        it = item_map[tid]
        q = it["question"]
        gold_page = it["gold_page"]

        print("=" * 70)
        print(f"{tid}: {q}")
        print(f"gold_page = {gold_page}")
        print("=" * 70)

        # 粗排 top-5：看粗排为什么把 gold 排在前面
        coarse = search(q, top_k=COARSE_TOP_K)
        print("【粗排 top-5】")
        for i, r in enumerate(coarse[:5], start=1):
            marker = " ★" if r.chunk.page_start <= gold_page <= r.chunk.page_end else ""
            print(f"  #{i}  score={r.score:.4f}  p{r.chunk.page_start}-{r.chunk.page_end}{marker}")
            print(f"      {r.chunk.text[:200]!r}")

        # 精排 top-8：看精排怎么排的
        # 传 final_k=20，拿到真实排名，只打印前 8
        fine = rerank(q, coarse, final_k=20)
        print()
        print("【精排 top-8（实际看 20，只打印 8）】")
        for i, r in enumerate(fine[:8], start=1):
            marker = " ★" if r.chunk.page_start <= gold_page <= r.chunk.page_end else ""
            print(f"  #{i}  score={r.score:.4f}  p{r.chunk.page_start}-{r.chunk.page_end}{marker}")

        # 找 gold chunk 在精排里的真实排名
        gold_rank = None
        gold_chunk = None
        for i, r in enumerate(fine, start=1):
            if r.chunk.page_start <= gold_page <= r.chunk.page_end:
                gold_rank = i
                gold_chunk = r.chunk
                break

        print()
        print(f"【gold chunk 在精排里的真实排名】: 第 {gold_rank} 名")
        if gold_chunk:
            # 判断是否超过 reranker 的 512 token 上限
            pair_text = q + " " + gold_chunk.text
            n_tokens = len(tok(pair_text, add_special_tokens=True)["input_ids"])
            warn = "  ⚠ 超过 512，会被 reranker 截断" if n_tokens > 512 else "  ✓ 未超过 512"
            print(f"【gold chunk 与 query 拼接后的 token 长度】: {n_tokens}{warn}")
            print(f"【gold chunk 原文（前 500 字符）】:")
            print(f"  {gold_chunk.text[:500]!r}")
        else:
            print("（gold chunk 不在精排 top-20 里，说明精排彻底把它推走了）")

        print()


if __name__ == "__main__":
    diagnose_remaining()