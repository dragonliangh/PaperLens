"""E4 诊断：打印英文可答题 vs 不可答题的 top_score 分布，找最优阈值。

【目的】
  当前英文阈值 0.3 是拍脑袋定的。需要看两类题的分数分布：
    - 可答题的最低分、中位数、最高分
    - 不可答题的最低分、中位数、最高分
  如果两类分布有「真空区」，阈值选在真空区中间即可。
  如果完全重叠，说明 0.3 附近没有区分度，需要另找突破点。

【为什么不跑全量评估】
  只需要 50 + 20 = 70 道题的 top_score，约 1-2 分钟。
  全量评估 23 分钟，还包含 Recall/MRR，对调阈值不是必需的。

【输出】
  1. 分数分位数（min / p25 / median / p75 / max）
  2. 阈值扫描表（0.15–0.50，步长 0.02），列出误拒数、漏网数、总错数
  3. 建议阈值（总错数最小的那个点）
"""
import json

from index_store import search
from rerank import rerank
from evaluate import load_eval_set, load_thresholds

EVAL_SET_PATH = "eval_set.json"
THRESHOLDS_PATH = "thresholds.json"
COARSE_TOP_K = 20
FINE_FINAL_K = 4


def diagnose_threshold_en():
    items = load_eval_set(EVAL_SET_PATH)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    en_items = [it for it in items if it["lang"] == "en"]

    answerable = [it for it in en_items if it["answerable"]]
    unanswerable = [it for it in en_items if not it["answerable"]]

    print("=" * 60)
    print("英文 top_score 分布诊断")
    print("=" * 60)

    # 收集可答题的 top_score
    ans_scores = []
    for it in answerable:
        coarse = search(it["question"], top_k=COARSE_TOP_K)
        fine = rerank(it["question"], coarse, final_k=FINE_FINAL_K)
        if fine:
            ans_scores.append((it["id"], fine[0].score))

    # 收集不可答题的 top_score，并记录 category
    unans_scores = []
    for it in unanswerable:
        coarse = search(it["question"], top_k=COARSE_TOP_K)
        fine = rerank(it["question"], coarse, final_k=FINE_FINAL_K)
        if fine:
            unans_scores.append((it["id"], it["category"], fine[0].score))

    def summarize(name, scores):
        vals = sorted(s for _, s in scores)
        if not vals:
            print(f"{name}: 空")
            return
        n = len(vals)
        print(f"{name}: n={n}")
        print(f"  min={vals[0]:.4f}  p25={vals[n//4]:.4f}  "
              f"median={vals[n//2]:.4f}  p75={vals[3*n//4]:.4f}  max={vals[-1]:.4f}")

    summarize("可答题", ans_scores)
    summarize("不可答题（全部）", [(i, s) for i, _, s in unans_scores])

    # 按 category 细分不可答题
    print()
    print("不可答题按 category 细分：")
    for cat in ["unrelated", "semi_related", "counterfactual"]:
        vals = sorted(s for _, c, s in unans_scores if c == cat)
        if vals:
            n = len(vals)
            print(f"  {cat:<16} n={n}  min={vals[0]:.4f}  median={vals[n//2]:.4f}  max={vals[-1]:.4f}")

    # 阈值扫描
    print()
    print("=" * 60)
    print("阈值扫描（目标：误拒数 + 漏网数 最小）")
    print("=" * 60)
    current = thresholds.get("en", thresholds["default"])
    print(f"当前英文阈值：{current}")
    print(f"{'阈值':<8} {'误拒':<8} {'漏网':<8} {'总错':<8}")
    print("-" * 40)

    best_th = current
    best_err = 999
    for th_int in range(15, 51, 2):
        th = th_int / 100
        miss = sum(1 for _, s in ans_scores if s < th)
        leak = sum(1 for _, _, s in unans_scores if s >= th)
        err = miss + leak
        marker = " ← 当前" if abs(th - current) < 0.01 else ""
        print(f"{th:<8.2f} {miss:<8} {leak:<8} {err:<8}{marker}")
        if err < best_err:
            best_err = err
            best_th = th

    print()
    print(f"建议阈值：{best_th:.2f}（总错 {best_err} 道，当前 {current} 总错 "
          f"{sum(1 for _, s in ans_scores if s < current) + sum(1 for _, _, s in unans_scores if s >= current)} 道）")

    # 打印边界案例：可答题最低 5 个、不可答题最高 5 个
    print()
    print("=" * 60)
    print("边界案例（可答题最低 5 个 vs 不可答题最高 5 个）")
    print("=" * 60)
    print("可答题（分数最低 5 个，最容易被误拒）：")
    for i, s in sorted(ans_scores, key=lambda x: x[1])[:5]:
        print(f"  {i}  score={s:.4f}")
    print("不可答题（分数最高 5 个，最容易漏网）：")
    for i, c, s in sorted(unans_scores, key=lambda x: -x[2])[:5]:
        print(f"  {i}  [{c}]  score={s:.4f}")


if __name__ == "__main__":
    diagnose_threshold_en()