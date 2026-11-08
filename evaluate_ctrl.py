"""对照组评估：评估 20 道事实型题（中英文各一版，共 40 条记录）。

【模块职责】
  读 ctrl_eval_set.json（40 条：20 英 + 20 中），对每道题跑「粗排 + 精排」，
  计算 Recall@5、MRR@20、Recall@4、MRR@4。
  与 evaluate.py 的区别：
    - 对照组没有不可答题，不测拒答指标。
    - 对照组按 type（T1–T5）分组统计，看「哪种题型最强」。
    - 输出写 ctrl_eval_report.json。

【在架构中的位置】
  第 7 块补充：主评估集（140 题）测「方法在混合题型上的整体表现」，
  对照组（20 道题，中英各一版）测「方法在擅长领域确实强」。两者对照，形成
  「主指标 + 边界分类 + 对照组」三层结构。

【遵循的契约】
  - 沿用第 7 块契约 15a–15e 的指标口径：
    * 粗排 Recall@5：top-20 中前 5 是否命中 gold_page。
    * 粗排 MRR@20：top-20 中第一个命中的排名倒数。
    * 精排 Recall@4：top-4 中是否命中。
    * 精排 MRR@4：top-4 中第一个命中的排名倒数。
    * 命中判定：gold_page ∈ [page_start, page_end]。
  - 中英文分开统计，不做综合平均。
  - 按 type 分组统计，看哪种题型更强。

【历史】
  - 新增于第 7 块补充（对照组评估）。
  - 修正 1：docstring 从「20 道题（en 10 + zh 10）」改为「20 道题（中英各一版，40 条记录）」。
  - 修正 2（本轮）：ctrl_eval_set.json 直接加 answerable / category 字段，
    load_ctrl_eval_set 退回纯 json.load，不在代码里补字段。

【依赖】
  - 标准库 json、os、datetime。
  - index_store、rerank、evaluate（复用 compute_metrics）。
"""
import json
import os
from datetime import datetime

from evaluate import compute_metrics

EVAL_SET_PATH = "ctrl_eval_set.json"
REPORT_PATH = "ctrl_eval_report.json"


def load_ctrl_eval_set(path: str) -> list[dict]:
    """读取对照组评估集 JSON。

    对照组 JSON 与主评估集结构一致，每条含：
      id / lang / type / question / answerable / category / gold_page / gold_keywords

    参数：
      path : ctrl_eval_set.json 的路径。

    返回：
      条目列表（40 条：20 英 + 20 中）。
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics_by_type(items: list[dict], thresholds: dict) -> dict:
    """按 type（T1–T5）分组，各自计算指标。

    参数：
      items      : 同语言的对照组条目。
      thresholds : 阈值字典（对照组没有不可答题，阈值不影响指标）。

    返回：
      dict，键为 type，值为该 type 的指标 dict。
    """
    by_type = {}
    for it in items:
        by_type.setdefault(it["type"], []).append(it)

    result = {}
    for t, sub_items in by_type.items():
        result[t] = compute_metrics(sub_items, thresholds)
    return result


def main():
    """跑对照组评估，打印表格 + 写 ctrl_eval_report.json。"""
    items = load_ctrl_eval_set(EVAL_SET_PATH)

    en_items = [it for it in items if it["lang"] == "en"]
    zh_items = [it for it in items if it["lang"] == "zh"]

    # 对照组没有不可答题，阈值随便传个默认值（不影响指标计算）
    thresholds = {"default": 0.0}

    print(f"对照组：{len(items)} 条（en={len(en_items)}, zh={len(zh_items)}）")
    print()

    en_metrics = compute_metrics(en_items, thresholds)
    zh_metrics = compute_metrics(zh_items, thresholds)

    print(f"{'指标':<28} {'EN':>10} {'ZH':>10}")
    print("-" * 50)
    rows = [
        ("可答题数", "answerable_count", 0),
        ("粗排 Recall@5", "coarse_recall_at_5", 4),
        ("粗排 MRR@20", "coarse_mrr_at_20", 4),
        ("精排 Recall@4", "fine_recall_at_4", 4),
        ("精排 MRR@4", "fine_mrr_at_4", 4),
    ]
    for label, key, fmt in rows:
        ev = en_metrics[key]
        zv = zh_metrics[key]
        if fmt:
            print(f"{label:<28} {ev:>10.4f} {zv:>10.4f}")
        else:
            print(f"{label:<28} {ev:>10d} {zv:>10d}")

    # 按 type 分组统计（只看英文，因为中英语义对应）
    print()
    print("按 type 分组（EN）：")
    print(f"{'type':<8} {'n':>4} {'Recall@5':>10} {'MRR@20':>10} {'Recall@4':>10} {'MRR@4':>10}")
    print("-" * 56)
    by_type = compute_metrics_by_type(en_items, thresholds)
    for t in ["T1", "T2", "T3", "T4", "T5"]:
        if t in by_type:
            m = by_type[t]
            print(f"{t:<8} {m['answerable_count']:>4} "
                  f"{m['coarse_recall_at_5']:>10.4f} {m['coarse_mrr_at_20']:>10.4f} "
                  f"{m['fine_recall_at_4']:>10.4f} {m['fine_mrr_at_4']:>10.4f}")

    report = {
        "timestamp": datetime.now().isoformat(),
        "en": en_metrics,
        "zh": zh_metrics,
        "en_by_type": by_type,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"报告已写入 {REPORT_PATH}")


if __name__ == "__main__":
    main()