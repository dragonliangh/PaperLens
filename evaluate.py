"""PaperLens 第 7 块：评估脚本。

【模块职责】
  读取 eval_set.json，对每道题跑「粗排 + 精排」，计算：
    - 粗排与精排的 Recall@4 / @5 及 MRR@4 / @5 / @20
    - 拒答准确率（不可答题中成功拒答的比例），按 category 细分
    - 排名分布直方图（证明为何选择 final_k=4）
  中英文分开统计，不做综合平均。

【遵循的契约（第 7 块）】
  - 契约 15a：eval_set.json 有 140 条，en 70 + zh 70，id 严格 001-070 连续。
  - 契约 15b：evaluate.py 跑完不报错，输出表格 + eval_report.json。
  - 契约 15c：en 和 zh 各自的 Recall / MRR / 拒答准确率都有数字。
  - 契约 15d：阈值从 thresholds.json 读，缺失抛异常。
  - 契约 15e：拒答准确率按 category（unrelated / semi_related / counterfactual）细分。

【遵循的设计决策】
  - 决策 5：阈值人工观察分布后定，不自动算。
  - 决策 6：阈值写在 thresholds.json，支持按语言分别设。
  - 决策 8：单文件，五个公开函数。
  - 额外诊断项：coarse_recall_at_20、fine_recall_at_20_gold_given、
    false_reject_rate、coarse_rank_histogram、n_fine_scored。

【指标口径】
  - 粗排：基于 search(top_k=20) 的结果，记录第一个命中 gold_page 的排名。
  - 精排：基于 rerank(final_k=20) 的全量结果，本地切片计算不同 K 的指标。
  - 命中判定：gold_page 落在 chunk 的 [page_start, page_end] 区间内。
  - 指标分母：统一使用可答题总数（n_ans），保证 A/B 对比绝对公平。
  - 可答题被拒答：单列 false_reject_rate。

【历史修改痕迹】
  - 第 7 块（第一轮）：创建 5 个公开函数、5 个诊断项、按 category 细分的拒答准确率。
  - 第 7 块（第二轮）：thresholds.json 改为字典，支持按语言设阈值。
  - 第 7 块（第三轮 / 本次）：指标口径重构 + 防卡死优化。
    * 修正了早期“粗排 Recall@5”与“精排 Recall@4”跨 K 对比的硬伤。
    * rerank 调用从 final_k=4 改为 final_k=20，本地切片支持同 K 对比。
    * 新增进度打印，解决跑长任务无反馈的问题。
    * 增加 torch.set_num_threads(4) 限制 CPU 占用，防止低配电脑 Swap 假死。
    * 新增排名分布直方图诊断，解决“为什么选 final_k=4”的验证盲区。
  - 第 7 块（第三轮补充 / 本次）：支持逐题明细与阈值参数化。
    * `compute_metrics` 额外收集 `_details`（逐题 rank/score/top1/是否被拒），
      由 main() pop 出来单独落盘，不污染报告字段。
    * 新增 `--report` / `--details` / `--zh-threshold` / `--en-threshold`。
      有了明细，zh 0.15 与 0.18 的对比、K=4 与 K=5 的分段表都可以在同一批
      分数上离线算，不需要把 140 题重跑两遍。

【对外暴露】
  - main()：跑完整评估，打印表格 + 写报告（可选落盘逐题明细）。

【依赖】
  - 标准库 argparse、json、os、datetime、time。
  - index_store、rerank、torch。
"""
import argparse
import json
import os
import time
from datetime import datetime
import torch

# 【核心逻辑 / 性能优化】限制 PyTorch 的线程数。
# 用户的物理机只有 4 核 8 线程，并且内存只有 8GB。
# 默认情况下 PyTorch 会尝试占满所有逻辑核，导致系统 UI 卡顿、引发内存 Swap 假死。
# 锁定 4 线程能让它在后台平稳跑完。
torch.set_num_threads(4)

from index_store import search
from rerank import rerank

# 文件路径：集中定义，便于测试时替换。
EVAL_SET_PATH = "eval_set.json"
THRESHOLDS_PATH = "thresholds.json"
REPORT_PATH = "eval_report.json"

# 检索参数：与 qa.py 保持一致。
COARSE_TOP_K = 20
# 【核心逻辑】为了保证评估的对比全面性，精排这里取全量 20 个结果。
# 生产环境 qa.py 里 final_k=4，但评估我们需要看它在 1..20 各段的分布。
FINE_FULL_K = 20

# 进度打印间隔
PROGRESS_EVERY = 5


def load_eval_set(path: str) -> list[dict]:
    """读取评估集 JSON。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_thresholds(path: str) -> dict:
    """读取拒答阈值字典。"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少 {path}，请先创建")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)["reject"]
    if isinstance(raw, dict):
        result = dict(raw)
    else:
        result = {"default": float(raw)}
    if "default" not in result:
        result["default"] = 0.3
    return result


def _get_threshold(thresholds: dict, lang: str) -> float:
    """按语言取阈值；lang 不在字典里时回退到 default。"""
    return thresholds.get(lang, thresholds["default"])


def _hit(chunk, gold_page: int) -> bool:
    """判断某个 chunk 是否命中 gold_page。区间判定，容忍跨页。"""
    if gold_page is None:
        return False
    return chunk.page_start <= gold_page <= chunk.page_end


def _first_hit_rank(results: list, gold_page: int) -> int | None:
    """返回第一个命中 gold_page 的 1-based 排名；未命中返回 None。"""
    for i, r in enumerate(results, start=1):
        if _hit(r.chunk, gold_page):
            return i
    return None


def _get_bin(rank: int | None) -> str:
    """将排名映射到区间桶，用于直方图统计。"""
    if rank is None:
        return "miss"
    if rank == 1:
        return "1"
    if rank == 2:
        return "2"
    if rank == 3:
        return "3"
    if rank == 4:
        return "4"
    if rank == 5:
        return "5"
    if 6 <= rank <= 10:
        return "6-10"
    if 11 <= rank <= 20:
        return "11-20"
    return ">20"


def _histogram(ranks: list[int | None]) -> dict:
    """生成排名分布直方图。"""
    hist = {b: 0 for b in ["1", "2", "3", "4", "5", "6-10", "11-20", "miss"]}
    for r in ranks:
        hist[_get_bin(r)] += 1
    return hist


def compute_metrics(items: list[dict], thresholds: dict, desc: str = "") -> dict:
    """对一组题（同语言）计算全部指标和诊断项。

    【第 7 块第三轮补充】返回值里带一个 `_details` 键：逐题明细
    （qid / lang / gold_page / 粗排名次与分数 / 精排名次与分数 / top1 分数 / 是否被拒）。
    它不是报告字段，main() 会把它 pop 出来单独落盘，用于：
      1. 复核 K=4 与 K=5 的截断差异（排名分段 + 分数分布）；
      2. 用同一批分数离线重算不同拒答阈值（zh 0.15 / 0.18），不必重跑。
    """
    answerable = [it for it in items if it["answerable"]]
    unanswerable = [it for it in items if not it["answerable"]]
    details: list[dict] = []

    # 累计器：粗排
    coarse_hits_4 = 0
    coarse_hits_5 = 0
    coarse_mrr_4_sum = 0.0
    coarse_mrr_5_sum = 0.0
    coarse_mrr_20_sum = 0.0
    coarse_ranks = []

    # 累计器：精排
    fine_hits_4 = 0
    fine_hits_5 = 0
    fine_mrr_4_sum = 0.0
    fine_mrr_5_sum = 0.0
    fine_mrr_20_sum = 0.0
    fine_ranks = []
    n_fine_scored = 0

    false_rejects = 0
    gold_in_coarse_and_fine = 0
    gold_in_coarse_lost_in_fine = 0

    t_start = time.time()
    total_ans = len(answerable)
    print(f"\n[{desc}] 开始评估可答题 ({total_ans} 道)...")

    for i, it in enumerate(answerable, start=1):
        q = it["question"]
        gold_page = it["gold_page"]
        threshold = _get_threshold(thresholds, it["lang"])

        # 1. 粗排
        coarse = search(q, top_k=COARSE_TOP_K)
        coarse_rank = _first_hit_rank(coarse, gold_page)
        coarse_ranks.append(coarse_rank)

        if coarse_rank is not None:
            coarse_mrr_20_sum += 1.0 / coarse_rank
            if coarse_rank <= 4:
                coarse_hits_4 += 1
                coarse_mrr_4_sum += 1.0 / coarse_rank
            if coarse_rank <= 5:
                coarse_hits_5 += 1
                coarse_mrr_5_sum += 1.0 / coarse_rank

        # 2. 精排
        fine_all = rerank(q, coarse, final_k=FINE_FULL_K)
        if fine_all:
            n_fine_scored += 1

        fine_rank = _first_hit_rank(fine_all, gold_page)
        fine_ranks.append(fine_rank)

        if fine_rank is not None:
            fine_mrr_20_sum += 1.0 / fine_rank
            if fine_rank <= 4:
                fine_hits_4 += 1
                fine_mrr_4_sum += 1.0 / fine_rank
            if fine_rank <= 5:
                fine_hits_5 += 1
                fine_mrr_5_sum += 1.0 / fine_rank

        # 诊断：粗排命中后，精排是否保住
        if coarse_rank is not None and coarse_rank <= COARSE_TOP_K:
            if fine_rank is not None:
                gold_in_coarse_and_fine += 1
            else:
                gold_in_coarse_lost_in_fine += 1

        top_score = fine_all[0].score if fine_all else 0.0
        if top_score < threshold:
            false_rejects += 1

        # 逐题明细（不进报告，main() 单独落盘）
        coarse_hit_chunk = coarse[coarse_rank - 1] if coarse_rank else None
        fine_hit_chunk = fine_all[fine_rank - 1] if fine_rank else None
        details.append({
            "id": it["id"],
            "lang": it["lang"],
            "answerable": True,
            "category": it.get("category"),
            "gold_page": gold_page,
            "coarse_rank": coarse_rank,
            "coarse_gold_score": round(coarse_hit_chunk.score, 6) if coarse_hit_chunk else None,
            "coarse_top1_score": round(coarse[0].score, 6) if coarse else None,
            "coarse_top1_page": coarse[0].chunk.page_start if coarse else None,
            "fine_rank": fine_rank,
            "fine_gold_score": round(fine_hit_chunk.score, 6) if fine_hit_chunk else None,
            "fine_top1_score": round(top_score, 6),
            "fine_top1_page": fine_all[0].chunk.page_start if fine_all else None,
            "threshold": threshold,
            "refused": top_score < threshold,
        })

        if i % PROGRESS_EVERY == 0 or i == total_ans:
            elapsed = time.time() - t_start
            print(f"  [{desc}] 可答题进度 {i}/{total_ans}，已用 {elapsed:.1f} 秒")

    # 不可答题进度
    total_unans = len(unanswerable)
    print(f"[{desc}] 开始评估不可答题 ({total_unans} 道)...")
    refusal_by_cat = {}

    for i, it in enumerate(unanswerable, start=1):
        q = it["question"]
        cat = it["category"]
        threshold = _get_threshold(thresholds, it["lang"])
        coarse = search(q, top_k=COARSE_TOP_K)
        fine = rerank(q, coarse, final_k=FINE_FULL_K)
        top_score = fine[0].score if fine else 0.0
        refused = top_score < threshold
        refusal_by_cat.setdefault(cat, {"total": 0, "refused": 0})
        refusal_by_cat[cat]["total"] += 1
        if refused:
            refusal_by_cat[cat]["refused"] += 1

        details.append({
            "id": it["id"],
            "lang": it["lang"],
            "answerable": False,
            "category": cat,
            "gold_page": None,
            "coarse_rank": None,
            "coarse_gold_score": None,
            "coarse_top1_score": round(coarse[0].score, 6) if coarse else None,
            "coarse_top1_page": coarse[0].chunk.page_start if coarse else None,
            "fine_rank": None,
            "fine_gold_score": None,
            "fine_top1_score": round(top_score, 6),
            "fine_top1_page": fine[0].chunk.page_start if fine else None,
            "threshold": threshold,
            "refused": refused,
        })

        if i % PROGRESS_EVERY == 0 or i == total_unans:
            elapsed = time.time() - t_start
            print(f"  [{desc}] 不可答题进度 {i}/{total_unans}，总计已用 {elapsed:.1f} 秒")

    n_ans = len(answerable)
    if n_ans == 0:
        return {}

    coarse_recall_4 = coarse_hits_4 / n_ans
    coarse_recall_5 = coarse_hits_5 / n_ans
    coarse_mrr_4 = coarse_mrr_4_sum / n_ans
    coarse_mrr_5 = coarse_mrr_5_sum / n_ans
    coarse_mrr_20 = coarse_mrr_20_sum / n_ans

    fine_recall_4 = fine_hits_4 / n_ans
    fine_recall_5 = fine_hits_5 / n_ans
    fine_mrr_4 = fine_mrr_4_sum / n_ans
    fine_mrr_5 = fine_mrr_5_sum / n_ans
    fine_mrr_20 = fine_mrr_20_sum / n_ans

    false_reject_rate = false_rejects / n_ans

    coarse_recall_20 = sum(
        1 for r in coarse_ranks if r is not None and r <= COARSE_TOP_K
    ) / n_ans

    total_gold_in_coarse = gold_in_coarse_and_fine + gold_in_coarse_lost_in_fine
    fine_recall_at_20_gold_given = (
        gold_in_coarse_and_fine / total_gold_in_coarse
        if total_gold_in_coarse else 0.0
    )

    refusal_accuracy = {
        cat: (d["refused"] / d["total"] if d["total"] else 0.0)
        for cat, d in refusal_by_cat.items()
    }
    n_unans = len(unanswerable)
    overall_refusal = (
        sum(d["refused"] for d in refusal_by_cat.values()) / n_unans
        if n_unans else 0.0
    )

    return {
        "answerable_count": n_ans,
        "unanswerable_count": n_unans,
        "coarse_recall_at_4": round(coarse_recall_4, 4),
        "coarse_recall_at_5": round(coarse_recall_5, 4),
        "coarse_mrr_at_4": round(coarse_mrr_4, 4),
        "coarse_mrr_at_5": round(coarse_mrr_5, 4),
        "coarse_mrr_at_20": round(coarse_mrr_20, 4),
        "fine_recall_at_4": round(fine_recall_4, 4),
        "fine_recall_at_5": round(fine_recall_5, 4),
        "fine_mrr_at_4": round(fine_mrr_4, 4),
        "fine_mrr_at_5": round(fine_mrr_5, 4),
        "fine_mrr_at_20": round(fine_mrr_20, 4),
        "refusal_accuracy": round(overall_refusal, 4),
        "refusal_accuracy_by_category": {
            k: round(v, 4) for k, v in refusal_accuracy.items()
        },
        "diagnostics": {
            "coarse_recall_at_20": round(coarse_recall_20, 4),
            "fine_recall_at_20_gold_given": round(fine_recall_at_20_gold_given, 4),
            "false_reject_rate": round(false_reject_rate, 4),
            "n_fine_scored": n_fine_scored,
            "coarse_rank_histogram": _histogram(coarse_ranks),
            "fine_rank_histogram": _histogram(fine_ranks),
        },
        "_details": details,
    }


def main():
    """跑完整评估，打印表格 + 写报告（可选落盘逐题明细）。"""
    parser = argparse.ArgumentParser(
        description="PaperLens 检索评估：粗排/精排分段指标 + 拒答准确率"
    )
    parser.add_argument(
        "--report", default=REPORT_PATH,
        help=f"报告输出路径（默认 {REPORT_PATH}）",
    )
    parser.add_argument(
        "--details", default=None,
        help="逐题明细的输出路径（JSON 数组）；不给则不落盘",
    )
    parser.add_argument(
        "--zh-threshold", type=float, default=None,
        help="覆盖 thresholds.json 里的 zh 拒答阈值（如 0.15 / 0.18）",
    )
    parser.add_argument(
        "--en-threshold", type=float, default=None,
        help="覆盖 thresholds.json 里的 en 拒答阈值",
    )
    args = parser.parse_args()

    items = load_eval_set(EVAL_SET_PATH)
    thresholds = load_thresholds(THRESHOLDS_PATH)
    if args.zh_threshold is not None:
        thresholds["zh"] = args.zh_threshold
    if args.en_threshold is not None:
        thresholds["en"] = args.en_threshold

    en_items = [it for it in items if it["lang"] == "en"]
    zh_items = [it for it in items if it["lang"] == "zh"]

    print(f"评估集：{len(items)} 条（en={len(en_items)}, zh={len(zh_items)}）")
    print(f"阈值：{thresholds}")

    en_metrics = compute_metrics(en_items, thresholds, desc="EN")
    zh_metrics = compute_metrics(zh_items, thresholds, desc="ZH")

    print("\n" + "=" * 60)
    print("核心指标（同 K 截断消融对比）")
    print("=" * 60)
    print(f"{'指标':<24} {'EN':>10} {'ZH':>10}")
    print("-" * 50)

    def print_row(label, en_key, zh_key):
        print(f"{label:<24} {en_metrics[en_key]:>10.4f} {zh_metrics[zh_key]:>10.4f}")

    print("【Top-4 截断（生产口径）】")
    print_row("粗排 Recall@4", "coarse_recall_at_4", "coarse_recall_at_4")
    print_row("精排 Recall@4", "fine_recall_at_4", "fine_recall_at_4")
    print_row("粗排 MRR@4", "coarse_mrr_at_4", "coarse_mrr_at_4")
    print_row("精排 MRR@4", "fine_mrr_at_4", "fine_mrr_at_4")

    print("\n【Top-5 截断（对照口径）】")
    print_row("粗排 Recall@5", "coarse_recall_at_5", "coarse_recall_at_5")
    print_row("精排 Recall@5", "fine_recall_at_5", "fine_recall_at_5")
    print_row("粗排 MRR@5", "coarse_mrr_at_5", "coarse_mrr_at_5")
    print_row("精排 MRR@5", "fine_mrr_at_5", "fine_mrr_at_5")

    print("\n【全量对比】")
    print_row("粗排 MRR@20", "coarse_mrr_at_20", "coarse_mrr_at_20")
    print_row("精排 MRR@20", "fine_mrr_at_20", "fine_mrr_at_20")
    print_row("拒答准确率", "refusal_accuracy", "refusal_accuracy")

    print("\n" + "=" * 60)
    print("排名分布直方图（证明为何 final_k=4）")
    print("=" * 60)
    bins = ["1", "2", "3", "4", "5", "6-10", "11-20", "miss"]
    en_c_hist = en_metrics["diagnostics"]["coarse_rank_histogram"]
    en_f_hist = en_metrics["diagnostics"]["fine_rank_histogram"]
    zh_c_hist = zh_metrics["diagnostics"]["coarse_rank_histogram"]
    zh_f_hist = zh_metrics["diagnostics"]["fine_rank_histogram"]

    print(f"{'区间':<8} | {'EN 粗排':<8} {'EN 精排':<8} | {'ZH 粗排':<8} {'ZH 精排':<8}")
    print("-" * 50)
    for b in bins:
        print(f"{b:<8} | {en_c_hist[b]:<8} {en_f_hist[b]:<8} | {zh_c_hist[b]:<8} {zh_f_hist[b]:<8}")

    print("\n" + "=" * 60)
    print("诊断项")
    print("=" * 60)
    print(f"{'指标':<34} {'EN':>10} {'ZH':>10}")
    print("-" * 56)

    diag_rows = [
        ("粗排 Recall@20（天花板）", "coarse_recall_at_20", 4),
        ("精排 Recall@4（给定粗排命中）", "fine_recall_at_20_gold_given", 4),
        ("误拒率（可答题被拒）", "false_reject_rate", 4),
        ("精排有效题数", "n_fine_scored", 0),
    ]
    for label, key, fmt in diag_rows:
        ev = en_metrics["diagnostics"][key]
        zv = zh_metrics["diagnostics"][key]
        if fmt:
            print(f"{label:<34} {ev:>10.4f} {zv:>10.4f}")
        else:
            print(f"{label:<34} {ev:>10d} {zv:>10d}")

    print("\n拒答准确率（按 category 细分）：")
    for cat in ["unrelated", "semi_related", "counterfactual"]:
        ev = en_metrics["refusal_accuracy_by_category"].get(cat, 0.0)
        zv = zh_metrics["refusal_accuracy_by_category"].get(cat, 0.0)
        print(f"  {cat:<16} EN={ev:.4f}  ZH={zv:.4f}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "thresholds_used": thresholds,
        "en": en_metrics,
        "zh": zh_metrics,
    }
    en_details = en_metrics.pop("_details", [])
    zh_details = zh_metrics.pop("_details", [])

    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n报告已写入 {args.report}")

    if args.details:
        all_details = en_details + zh_details
        with open(args.details, "w", encoding="utf-8") as f:
            json.dump(all_details, f, ensure_ascii=False, indent=2)
        print(f"逐题明细已写入 {args.details}（{len(all_details)} 条）")


if __name__ == "__main__":
    main()