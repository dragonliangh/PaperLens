"""排查 4 道"被精排推走"的题。

【背景】
  诊断发现 4 道题（en_016、en_029、en_042、en_044）在粗排里排第 2-4 名，
  但精排把它们推到了第 7 名及以后。这不是"精排没排好"，而是"精排主动推走了
  粗排已经排好的答案"。

【目的】
  对这 4 道题，打印：
    1. 粗排 top-5 的 chunk（前 200 字符），看粗排为什么排得对。
    2. 精排 top-8 的排名和分数，看精排怎么推走的。
    3. gold_page 对应的 chunk 原文前 500 字符，人工判断是：
       - 截断问题（chunk 太长，答案在末尾被 reranker 截掉）
       - 模型判断错（chunk 不长，但 reranker 打分就是低）
       - 语义歧义（question 有多种解释，reranker 选了另一种）

【输出】
  每道题打印三段信息，人工看后能给出结论。
"""
from index_store import search
from rerank import rerank
from evaluate import _first_hit_rank, load_eval_set
from ingest import _get_tokenizer

EVAL_SET_PATH = "eval_set.json"
COARSE_TOP_K = 20
TARGET_IDS = ["en_016", "en_029", "en_042", "en_044"]


def diagnose_pushed_away():
    items = load_eval_set(EVAL_SET_PATH)
    item_map = {it["id"]: it for it in items if it["id"] in TARGET_IDS}

    tok = _get_tokenizer()

    for tid in TARGET_IDS:
        it = item_map[tid]
        q = it["question"]
        gold_page = it["gold_page"]

        print("=" * 70)
        print(f"{tid}: {q}")
        print(f"gold_page = {gold_page}")
        print("=" * 70)

        # 粗排 top-5
        coarse = search(q, top_k=COARSE_TOP_K)
        print("【粗排 top-5】")
        for i, r in enumerate(coarse[:5], start=1):
            marker = " ★" if r.chunk.page_start <= gold_page <= r.chunk.page_end else ""
            print(f"  #{i}  score={r.score:.4f}  p{r.chunk.page_start}-{r.chunk.page_end}{marker}")
            print(f"      {r.chunk.text[:200]!r}")

        # 精排 top-8
        fine = rerank(q, coarse, final_k=20)  # 传 20，看真实排名
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
            # 计算 gold chunk 和 query 拼接后的 token 长度
            pair_text = q + " " + gold_chunk.text
            n_tokens = len(tok(pair_text, add_special_tokens=True)["input_ids"])
            print(f"【gold chunk 与 query 拼接后的 token 长度】: {n_tokens}"
                  f"{'  ⚠ 超过 512，会被 reranker 截断' if n_tokens > 512 else '  ✓ 未超过 512'}")
            print(f"【gold chunk 原文（前 500 字符）】:")
            print(f"  {gold_chunk.text[:500]!r}")
        else:
            print("（gold chunk 不在精排 top-20 里）")

        print()


if __name__ == "__main__":
    diagnose_pushed_away()