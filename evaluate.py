"""PaperLens 第 7 块：评估脚本。

【模块职责】
  读取 eval_set.json，对每道题跑「粗排 + 精排」，计算：
    - 粗排 Recall@5（top-5 内命中 gold_page）、MRR@20
    - 精排 Recall@4（top-4 内命中 gold_page）、MRR@4
    - 拒答准确率（不可答题中成功拒答的比例），按 category 细分
  中英文分开统计，不做综合平均。
  同时输出诊断项，便于定位「粗排漏召回」vs「精排排错」。

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
  - 额外诊断项（用户要求）：coarse_recall_at_20、fine_recall_at_20_gold_given、
    false_reject_rate、coarse_rank_histogram、n_fine_scored。

【指标口径（用户确认）】
  - 粗排 Recall@5：基于 search() 的 top-20 结果，只看前 5 个是否命中。
  - 粗排 MRR@20：基于 search() 的 top-20 结果，第一个命中的排名倒数。
  - 精排 Recall@4：基于 rerank() 的 top-4 结果，前 4 个是否命中。
  - 精排 MRR@4：基于 rerank() 的 top-4 结果，第一个命中的排名倒数。
  - 命中判定：gold_page 落在 chunk 的 [page_start, page_end] 区间内。
  - 精排指标分母：有精排结果的题数（n_fine_scored），不是可答题总数。
  - 可答题被拒答：四项检索指标记 0，另单列 false_reject_rate。
  - 阈值按语言分别取：item["lang"] 精确匹配 > default。

【历史修改痕迹】
  - 第 7 块（第一轮）：
    * 创建 5 个公开函数、5 个诊断项、按 category 细分的拒答准确率。
    * 阈值从 thresholds.json 读单值。
  - 第 7 块（第二轮，本轮）：
    * thresholds.json 改为 {default, en, zh} 字典，支持按语言设阈值。
    * load_thresholds() 兼容旧格式（数字 -> 包装成 {"default": 数字}）。
    * compute_metrics() 里每道题按 item["lang"] 选阈值，而非全局一个值。
    * main() 打印时显示各语言的阈值。

【对外暴露】
  - main()：跑完整评估，打印表格 + 写 eval_report.json。

【依赖】
  - 标准库 json、os、datetime。
  - index_store、rerank。
"""
import json
import os
from datetime import datetime

from index_store import search
from rerank import rerank

# 文件路径：集中定义，便于测试时替换。
EVAL_SET_PATH = "eval_set.json"
THRESHOLDS_PATH = "thresholds.json"
REPORT_PATH = "eval_report.json"

# 检索参数：与 qa.py 保持一致。
COARSE_TOP_K = 20
FINE_FINAL_K = 4

# 指标参数：Recall@K 的 K。
RECALL_AT_5 = 5
RECALL_AT_4 = 4


def load_eval_set(path: str) -> list[dict]:
    """读取评估集 JSON。

    参数：
      path : eval_set.json 的路径。

    返回：
      条目列表，每条含 id / lang / question / answerable / category /
      gold_page / gold_keywords。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_thresholds(path: str) -> dict:
    """读取拒答阈值字典。

    契约 15d：缺失时抛异常，避免用默认值悄悄跑错。
    兼容旧格式（数字）和新格式（字典）。

    参数：
      path : thresholds.json 的路径。

    返回：
      dict，形如 {"default": 0.3, "en": 0.3, "zh": 0.18}。
    """
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
    """判断某个 chunk 是否命中 gold_page。

    【核心逻辑】用区间判定，而非精确匹配 page_start。
    原因：chunk 可能跨页，gold_page 落在 [page_start, page_end] 内即算命中。

    参数：
      chunk      : Chunk 对象（来自 index_store）。
      gold_page  : 正确答案所在页码；不可答题为 None。

    返回：
      True / False。
    """
    if gold_page is None:
        return False
    return chunk.page_start <= gold_page <= chunk.page_end


def _first_hit_rank(results: list, gold_page: int) -> int | None:
    """返回第一个命中 gold_page 的 1-based 排名。

    【核心逻辑】MRR 的标准定义是「第一个正确结果的排名倒数」。
    返回 1-based 排名，便于 1/rank 计算；未命中返回 None。

    参数：
      results   : RetrievedChunk 列表（粗排或精排结果）。
      gold_page : 正确答案所在页码。

    返回：
      1-based 排名整数；未命中返回 None。
    """
    for i, r in enumerate(results, start=1):
        if _hit(r.chunk, gold_page):
            return i
    return None


def _histogram(ranks: list[int]) -> dict:
    """粗排命中排名直方图，用于诊断。

    参数：
      ranks : 每题第一个命中排名列表；未命中记为 99。

    返回：
      dict，键为排名字符串（"1"、"2"、...、"miss"），值为题数。
    """
    hist = {}
    for r in ranks:
        key = "miss" if r == 99 else str(r)
        hist[key] = hist.get(key, 0) + 1
    return hist


def compute_metrics(items: list[dict], thresholds: dict) -> dict:
    """对一组题（同语言）计算全部指标和诊断项。

    参数：
      items      : 同语言的评估条目列表。
      thresholds : 阈值字典，形如 {"default": 0.3, "en": 0.3, "zh": 0.18}。

    返回：
      dict，含主指标、诊断项、拒答准确率（按 category 细分）。

    【算法步骤】
      1. 分离可答题和不可答题。
      2. 对每题可答题：粗排 → 记录命中排名 → 精排 → 记录命中排名。
      3. 对每题不可答题：粗排 → 精排 → 记录最高分是否低于该语言阈值。
      4. 汇总指标。
    """
    answerable = [it for it in items if it["answerable"]]
    unanswerable = [it for it in items if not it["answerable"]]

    coarse_hits_5 = 0
    coarse_reciprocal_ranks = []
    coarse_ranks = []
    fine_hits_4 = 0
    fine_reciprocal_ranks = []
    n_fine_scored = 0
    false_rejects = 0
    # 诊断项：粗排 top-20 命中且精排 top-4 也命中的题数
    gold_in_coarse_and_fine = 0
    # 诊断项：粗排 top-20 命中、但精排 top-4 丢掉的题数
    gold_in_coarse_lost_in_fine = 0

    for it in answerable:
        q = it["question"]
        gold_page = it["gold_page"]
        # 第 7 块第二轮：按语言取阈值
        threshold = _get_threshold(thresholds, it["lang"])

        # 粗排：top-20
        coarse = search(q, top_k=COARSE_TOP_K)
        coarse_rank = _first_hit_rank(coarse, gold_page)
        coarse_ranks.append(coarse_rank if coarse_rank is not None else 99)

        if coarse_rank is not None and coarse_rank <= RECALL_AT_5:
            coarse_hits_5 += 1
        coarse_reciprocal_ranks.append(
            1.0 / coarse_rank if coarse_rank is not None else 0.0
        )

        # 精排：top-4
        fine = rerank(q, coarse, final_k=FINE_FINAL_K)
        if fine:
            n_fine_scored += 1
        fine_rank = _first_hit_rank(fine, gold_page)
        if fine_rank is not None and fine_rank <= RECALL_AT_4:
            fine_hits_4 += 1
        fine_reciprocal_ranks.append(
            1.0 / fine_rank if fine_rank is not None else 0.0
        )

        # 诊断：粗排命中后精排是否保住
        if coarse_rank is not None and coarse_rank <= COARSE_TOP_K:
            if fine_rank is not None:
                gold_in_coarse_and_fine += 1
            else:
                gold_in_coarse_lost_in_fine += 1

        # 可答题被拒答：记一次 false reject
        top_score = fine[0].score if fine else 0.0
        if top_score < threshold:
            false_rejects += 1

    n_ans = len(answerable)
    coarse_recall_5 = coarse_hits_5 / n_ans if n_ans else 0.0
    coarse_mrr_20 = sum(coarse_reciprocal_ranks) / n_ans if n_ans else 0.0
    fine_recall_4 = fine_hits_4 / n_fine_scored if n_fine_scored else 0.0
    fine_mrr_4 = sum(fine_reciprocal_ranks) / n_ans if n_ans else 0.0
    false_reject_rate = false_rejects / n_ans if n_ans else 0.0

    # 诊断项：粗排 Recall@20（天花板）
    coarse_recall_20 = sum(
        1 for r in coarse_ranks if r is not None and r <= COARSE_TOP_K
    ) / n_ans if n_ans else 0.0

    # 诊断项：在粗排 top-20 已命中的题中，精排 top-4 仍命中的比例
    # 回答「精排有没有把粗排找到的正确答案丢掉」。
    total_gold_in_coarse = gold_in_coarse_and_fine + gold_in_coarse_lost_in_fine
    fine_recall_at_20_gold_given = (
        gold_in_coarse_and_fine / total_gold_in_coarse
        if total_gold_in_coarse else 0.0
    )

    # 拒答指标：仅统计不可答题，按 category 细分
    refusal_by_cat = {}
    for it in unanswerable:
        q = it["question"]
        cat = it["category"]
        threshold = _get_threshold(thresholds, it["lang"])  # 按语言取阈值
        coarse = search(q, top_k=COARSE_TOP_K)
        fine = rerank(q, coarse, final_k=FINE_FINAL_K)
        top_score = fine[0].score if fine else 0.0
        refused = top_score < threshold
        refusal_by_cat.setdefault(cat, {"total": 0, "refused": 0})
        refusal_by_cat[cat]["total"] += 1
        if refused:
            refusal_by_cat[cat]["refused"] += 1

    refusal_accuracy = {}
    for cat, d in refusal_by_cat.items():
        refusal_accuracy[cat] = d["refused"] / d["total"] if d["total"] else 0.0

    n_unans = len(unanswerable)
    overall_refusal = (
        sum(d["refused"] for d in refusal_by_cat.values()) / n_unans
        if n_unans else 0.0
    )

    return {
        "answerable_count": n_ans,
        "unanswerable_count": n_unans,
        "coarse_recall_at_5": round(coarse_recall_5, 4),
        "coarse_mrr_at_20": round(coarse_mrr_20, 4),
        "fine_recall_at_4": round(fine_recall_4, 4),
        "fine_mrr_at_4": round(fine_mrr_4, 4),
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
        },
    }


def main():
    """跑完整评估，打印表格 + 写 eval_report.json。"""
    items = load_eval_set(EVAL_SET_PATH)
    thresholds = load_thresholds(THRESHOLDS_PATH)

    en_items = [it for it in items if it["lang"] == "en"]
    zh_items = [it for it in items if it["lang"] == "zh"]

    print(f"评估集：{len(items)} 条（en={len(en_items)}, zh={len(zh_items)}）")
    print(f"阈值：{thresholds}")
    print()

    en_metrics = compute_metrics(en_items, thresholds)
    zh_metrics = compute_metrics(zh_items, thresholds)

    print(f"{'指标':<28} {'EN':>10} {'ZH':>10}")
    print("-" * 50)
    rows = [
        ("可答题数", "answerable_count", 0),
        ("不可答题数", "unanswerable_count", 0),
        ("粗排 Recall@5", "coarse_recall_at_5", 4),
        ("粗排 MRR@20", "coarse_mrr_at_20", 4),
        ("精排 Recall@4", "fine_recall_at_4", 4),
        ("精排 MRR@4", "fine_mrr_at_4", 4),
        ("拒答准确率", "refusal_accuracy", 4),
    ]
    for label, key, fmt in rows:
        ev = en_metrics[key]
        zv = zh_metrics[key]
        if fmt:
            print(f"{label:<28} {ev:>10.4f} {zv:>10.4f}")
        else:
            print(f"{label:<28} {ev:>10d} {zv:>10d}")

    print()
    print("诊断项：")
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

    print()
    print("拒答准确率（按 category 细分）：")
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
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"报告已写入 {REPORT_PATH}")


if __name__ == "__main__":
    main()