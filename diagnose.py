"""诊断脚本：找出"粗排命中但精排没进 top-4"的题，看它们在精排里排第几。

【目的】
  直接回答核心问题：丢掉的 15% 答案，在精排里是排第 5、第 6，还是排第 10？
  这决定了 final_k 调到 6 是否有意义。

【为什么不跑全量评估】
  全量评估 23 分钟。这里只对 50 道英文可答题跑粗排 + 精排，约 1-2 分钟。
  拿到分布后，再决定实验方向。

【遵循】
  只读 search / rerank / load_eval_set，不修改任何生产代码。

【历史修正】
  - 初版打印"粗排命中"用了三层嵌套减法，实际算出来是 total_lost（12），
    不是粗排命中数（44）。现改为显式计数器 coarse_hit_count。
"""
import json

from index_store import search
from rerank import rerank
from evaluate import _first_hit_rank, load_eval_set

EVAL_SET_PATH = "eval_set.json"
COARSE_TOP_K = 20
FINE_FINAL_K = 4
RECALL_AT_4 = 4


def diagnose_lost_answers():
    items = load_eval_set(EVAL_SET_PATH)
    en_answerable = [it for it in items if it["lang"] == "en" and it["answerable"]]

    lost_ranks = []
    total_ans = len(en_answerable)
    total_lost = 0
    # 【修正点】显式计数器：粗排 top-20 命中的题数。
    coarse_hit_count = 0

    for it in en_answerable:
        q = it["question"]
        gold_page = it["gold_page"]

        # 粗排
        coarse = search(q, top_k=COARSE_TOP_K)
        coarse_rank = _first_hit_rank(coarse, gold_page)
        if coarse_rank is None or coarse_rank > COARSE_TOP_K:
            continue  # 粗排都没找到，不计入"精排丢失"的统计
        # 【修正点】粗排命中，计数 +1
        coarse_hit_count += 1

        # 精排（这里送 top-8，看看丢掉的题能不能被救回来）
        fine = rerank(q, coarse, final_k=8)
        fine_rank = _first_hit_rank(fine, gold_page)

        if fine_rank is None or fine_rank > RECALL_AT_4:
            total_lost += 1
            lost_ranks.append({
                "id": it["id"],
                "coarse_rank": coarse_rank,
                "fine_rank": fine_rank if fine_rank is not None else ">8",
            })

    print("=" * 60)
    print(f"英文可答题总数：{total_ans}")
    # 【修正点】用显式计数器打印，而不是嵌套减法
    print(f"粗排命中（在 top-20 内）：{coarse_hit_count} 道")
    print(f"粗排命中但精排 top-4 未命中：{total_lost} 道")
    print("=" * 60)

    # 统计：如果 final_k 调到 6 或 8，能救回几道？
    can_save_6 = sum(1 for r in lost_ranks if isinstance(r["fine_rank"], int) and r["fine_rank"] <= 6)
    can_save_8 = sum(1 for r in lost_ranks if isinstance(r["fine_rank"], int) and r["fine_rank"] <= 8)
    beyond_8 = sum(1 for r in lost_ranks if r["fine_rank"] == ">8" or (isinstance(r["fine_rank"], int) and r["fine_rank"] > 8))

    print(f"如果 final_k 调到 6：能救回 {can_save_6} 道，仍丢失 {total_lost - can_save_6} 道")
    print(f"如果 final_k 调到 8：能救回 {can_save_8} 道，仍丢失 {total_lost - can_save_8} 道")
    print(f"排第 9 名以后（无论如何救不回）：{beyond_8} 道")
    print()

    print("明细（粗排命中但精排未进 top-4 的题）：")
    print(f"{'id':<10} {'粗排排名':<10} {'精排排名':<10}")
    for r in lost_ranks:
        print(f"{r['id']:<10} {r['coarse_rank']:<10} {r['fine_rank']:<10}")


if __name__ == "__main__":
    diagnose_lost_answers()