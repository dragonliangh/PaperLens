"""PaperLens 阶段 1：消融实验（有/无 Cross-Encoder 重排）。

【模块职责】
  量化第 6 块重排模块的贡献：同一套评估集、同一个粗排候选池，
  对比「带重排」与「不带重排」的精排指标（Recall@4 / MRR@4），
  产出可直接写进 README 的 A/B 数字与每题明细。

【在架构中的位置】
  第 7 块之后的独立分析脚本，不属于生产链路。
    - 上游依赖：index_store.search()（粗排）、rerank.rerank()（精排）、
      evaluate.load_eval_set / _first_hit_rank（复用口径，不复制，避免两套定义漂移）。
    - 下游被依赖：无。README「消融实验」一节的数字来源。
    - 明确不修改：evaluate.py / qa.py / rerank.py / index_store.py / ingest.py。

【类级】
  本文件无类定义，全部为模块级函数（纯函数脚本，与 evaluate.py 同风格）。

【对外暴露】
  - main() -> None ：跑完整消融，打印 A/B 对比表 + 写 ablation_report.json。

【遵循的契约（阶段 1）】
  - 1.4：A 组 = search(top_k=20) → rerank(final_k=4)；
         B 组 = search(top_k=20) 取前 4。
  - 1.4 B1：B 组取同一 top_k=20 候选池的前 4，保证唯一变量是「要不要重排」。
  - 1.4 B2：输出每题明细（with_rerank / without_rerank 的 hit 与 rank）。
  - 1.5：主指标 Recall@4 / MRR@4，中英文分开；辅助粗排 Recall@5 / MRR@20。
  - 1.6：空索引时照常跑完并写报告，不崩；空 eval_set 由 load_eval_set 抛异常。
  - 1.7：rerank 异常向上抛，不吞（不做降级——降级是阶段 4）。
  - 契约缺口确认：只对 answerable=true 的题做消融（en 50 + zh 50 = 100 条）。

【遵循的设计决策】
  - D1：search 每题只调 1 次，A/B 共用同一候选池（干净消融 + 省一半检索时间）。
  - D2：B 组用 coarse[:4] 切片，不重跑 search(top_k=4)。
  - D3：不复用 compute_metrics（它内部又调一次 search+rerank，会翻倍耗时）。
  - D4：每题存 a_rank / b_rank / coarse_rank 三个整数排名，信息量最大。
  - D5：每 10 题打一行进度（总耗时约 11 分钟）。
  - D7：rerank 异常直接抛，不伪装成「未命中」。

【依赖】
  标准库 json / time / datetime；index_store.search；rerank.rerank；
  evaluate 的 load_eval_set / _first_hit_rank（纯函数，不引入模型加载）。
"""
import json
import time
from datetime import datetime

from index_store import search
from rerank import rerank
from evaluate import load_eval_set, _first_hit_rank

# 文件路径：集中定义，便于测试时替换。
EVAL_SET_PATH = "eval_set.json"
REPORT_PATH = "ablation_report.json"

# 口径参数：与 evaluate.py 保持一致，便于两套报告交叉核对。
COARSE_TOP_K = 20     # 粗排候选数
FINAL_K = 4           # 精排返回数（B 组也取前 4，保证 A/B 同量）
RECALL_AT = 4         # 精排 Recall@K 的 K
RECALL_AT_5 = 5       # 粗排 Recall@5 的 K（辅助指标，与主评估口径一致）
PROGRESS_EVERY = 10   # 每 N 题打一行进度


def _aggregate(records: list[dict]) -> dict:
    """把一组题（同语言）的每题记录聚合成指标 dict。

    【方法契约】
      输入：records = [{id, lang, gold_page, with_rerank:{rank,hit},
                        without_rerank:{rank,hit}, coarse_rank}, ...]
      输出：{answerable_count, with_rerank:{recall_at_4,mrr_at_4},
             without_rerank:{...}, coarse_baseline:{recall_at_5,mrr_at_20}}
      不修改输入；纯函数，可在测试中单独调用。

    【算法】
      Recall@4 = 命中题数 / 总题数（rank 非 None 即命中，因为 rank 只在 1..4）
      MRR@4    = sum(1/rank if rank else 0) / 总题数（标准定义，未命中记 0）
    """
    n = len(records)
    # 【边界】空组防御：若 eval_set 结构异常导致某语言无条目，避免 ZeroDivisionError。
    # 契约 1.6 只承诺「空索引不崩」，此处是对「空组」的额外兜底，成本 2 行。
    if n == 0:
        zero = {"recall_at_4": 0.0, "mrr_at_4": 0.0}
        return {
            "answerable_count": 0,
            "with_rerank": dict(zero),
            "without_rerank": dict(zero),
            "coarse_baseline": {"recall_at_5": 0.0, "mrr_at_20": 0.0},
        }

    # 【核心逻辑】A 组（带重排）：rank 为 1..4 或 None
    a_hits = sum(1 for r in records if r["with_rerank"]["rank"] is not None)
    a_mrr = sum(
        1.0 / r["with_rerank"]["rank"] if r["with_rerank"]["rank"] is not None else 0.0
        for r in records
    ) / n

    # 【核心逻辑】B 组（不重排，取粗排前 4）：rank 为 1..4 或 None
    b_hits = sum(1 for r in records if r["without_rerank"]["rank"] is not None)
    b_mrr = sum(
        1.0 / r["without_rerank"]["rank"] if r["without_rerank"]["rank"] is not None else 0.0
        for r in records
    ) / n

    # 【核心逻辑】粗排辅助指标（A/B 共用）：coarse_rank 可能是 1..20 或 None
    coarse_hits_5 = sum(
        1 for r in records if r["coarse_rank"] is not None and r["coarse_rank"] <= RECALL_AT_5
    )
    coarse_mrr_20 = sum(
        1.0 / r["coarse_rank"] if r["coarse_rank"] is not None else 0.0
        for r in records
    ) / n

    return {
        "answerable_count": n,
        "with_rerank": {
            "recall_at_4": round(a_hits / n, 4),
            "mrr_at_4": round(a_mrr, 4),
        },
        "without_rerank": {
            "recall_at_4": round(b_hits / n, 4),
            "mrr_at_4": round(b_mrr, 4),
        },
        "coarse_baseline": {
            "recall_at_5": round(coarse_hits_5 / n, 4),
            "mrr_at_20": round(coarse_mrr_20, 4),
        },
    }


def _diff(records: list[dict]) -> dict:
    """计算「重排救回」与「重排推走」的题号列表。

    【方法契约】
      a_hit_b_miss：A 组命中且 B 组未命中 → 重排的正面贡献。
      b_hit_a_miss：B 组命中且 A 组未命中 → 重排的负面贡献（推走）。
      只统计 Recall@4 口径（rank 非 None 即命中），因为两组 rank 都 ≤ 4。

    【用途】
      README「消融实验」可写：「重排救回 X 道，推走 Y 道，净收益 X-Y」，
      并附具体题号，面试时可举例。
    """
    a_hit_b_miss = []
    b_hit_a_miss = []
    for r in records:
        a_hit = r["with_rerank"]["rank"] is not None
        b_hit = r["without_rerank"]["rank"] is not None
        if a_hit and not b_hit:
            a_hit_b_miss.append(r["id"])
        elif b_hit and not a_hit:
            b_hit_a_miss.append(r["id"])
    return {"a_hit_b_miss": a_hit_b_miss, "b_hit_a_miss": b_hit_a_miss}


def main() -> None:
    """跑完整消融，打印 A/B 对比表 + 写 ablation_report.json。"""
    items = load_eval_set(EVAL_SET_PATH)
    # 【核心逻辑 / 契约缺口确认】只对可答题做消融：
    # 不可答题 gold_page=None，_hit 恒 False，会把 Recall/MRR 分子分母同时拉低，
    # 语义上「重排对不可答题有没有用」也没有意义。
    answerable = [it for it in items if it["answerable"]]
    total = len(answerable)

    print(f"消融评估集：{total} 道可答题（原 {len(items)} 条，已过滤不可答题）")
    print(
        f"口径：A=search(top_k={COARSE_TOP_K})→rerank(final_k={FINAL_K})；"
        f"B=search(top_k={COARSE_TOP_K})→前{FINAL_K}"
    )
    print()

    records: list[dict] = []
    t_start = time.time()

    for i, it in enumerate(answerable, start=1):
        q = it["question"]
        gold_page = it["gold_page"]

        # 【核心逻辑 / D1】粗排只调一次，A/B 共用同一 20 候选池。
        coarse = search(q, top_k=COARSE_TOP_K)
        coarse_rank = _first_hit_rank(coarse, gold_page)

        # A 组：走重排
        # 【边界 / D7】rerank 异常向上抛，不吞——消融不做降级。
        fine = rerank(q, coarse, final_k=FINAL_K)
        a_rank = _first_hit_rank(fine, gold_page)

        # B 组：不重排，取粗排前 4。
        # 【D2】切片不重跑 search，保证与 A 组共用候选池（契约 B1）。
        b_rank = _first_hit_rank(coarse[:FINAL_K], gold_page)

        records.append(
            {
                "id": it["id"],
                "lang": it["lang"],
                "gold_page": gold_page,
                "with_rerank": {
                    "rank": a_rank,
                    "hit": a_rank is not None,  # rank 只可能是 1..4，命中即 ≤4
                },
                "without_rerank": {
                    "rank": b_rank,
                    "hit": b_rank is not None,
                },
                "coarse_rank": coarse_rank,
            }
        )

        # 【D5】每 10 题打进度：总耗时约 11 分钟，无进度用户会以为卡死。
        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"  进度 {i}/{total}，已用 {time.time() - t_start:.1f} 秒")

    en_records = [r for r in records if r["lang"] == "en"]
    zh_records = [r for r in records if r["lang"] == "zh"]
    en_metrics = _aggregate(en_records)
    zh_metrics = _aggregate(zh_records)

    # -----------------------------------------------------------------------
    # 打印 A/B 对比表
    # -----------------------------------------------------------------------
    print()
    print(f"{'指标':<32} {'EN':>10} {'ZH':>10}")
    print("-" * 56)

    def _row(label: str, en_val, zh_val, fmt: str = ".4f") -> None:
        # 【语句级】统一格式化，整数位用 d，浮点用 .4f，避免手写重复。
        if fmt == "d":
            print(f"{label:<32} {en_val:>10d} {zh_val:>10d}")
        else:
            print(f"{label:<32} {en_val:>10{fmt}} {zh_val:>10{fmt}}")

    _row("可答题数", en_metrics["answerable_count"], zh_metrics["answerable_count"], "d")
    _row(
        "粗排 Recall@5（A/B共用）",
        en_metrics["coarse_baseline"]["recall_at_5"],
        zh_metrics["coarse_baseline"]["recall_at_5"],
    )
    _row(
        "粗排 MRR@20（A/B共用）",
        en_metrics["coarse_baseline"]["mrr_at_20"],
        zh_metrics["coarse_baseline"]["mrr_at_20"],
    )
    _row(
        "A 组 Recall@4（带重排）",
        en_metrics["with_rerank"]["recall_at_4"],
        zh_metrics["with_rerank"]["recall_at_4"],
    )
    _row(
        "A 组 MRR@4",
        en_metrics["with_rerank"]["mrr_at_4"],
        zh_metrics["with_rerank"]["mrr_at_4"],
    )
    _row(
        "B 组 Recall@4（不重排）",
        en_metrics["without_rerank"]["recall_at_4"],
        zh_metrics["without_rerank"]["recall_at_4"],
    )
    _row(
        "B 组 MRR@4",
        en_metrics["without_rerank"]["mrr_at_4"],
        zh_metrics["without_rerank"]["mrr_at_4"],
    )
    # 【核心逻辑】Δ = A - B，正数表示重排有正面贡献。
    _row(
        "Δ Recall@4（A-B）",
        round(en_metrics["with_rerank"]["recall_at_4"]
              - en_metrics["without_rerank"]["recall_at_4"], 4),
        round(zh_metrics["with_rerank"]["recall_at_4"]
              - zh_metrics["without_rerank"]["recall_at_4"], 4),
    )
    _row(
        "Δ MRR@4（A-B）",
        round(en_metrics["with_rerank"]["mrr_at_4"]
              - en_metrics["without_rerank"]["mrr_at_4"], 4),
        round(zh_metrics["with_rerank"]["mrr_at_4"]
              - zh_metrics["without_rerank"]["mrr_at_4"], 4),
    )

    # -----------------------------------------------------------------------
    # 打印 diff 明细（B2 要求，面试可举具体例子）
    # -----------------------------------------------------------------------
    en_diff = _diff(en_records)
    zh_diff = _diff(zh_records)

    print()
    print("重排救回（A 命中 B 未命中）：")
    print(f"  EN: {en_diff['a_hit_b_miss']}  (n={len(en_diff['a_hit_b_miss'])})")
    print(f"  ZH: {zh_diff['a_hit_b_miss']}  (n={len(zh_diff['a_hit_b_miss'])})")
    print()
    print("重排推走（B 命中 A 未命中）：")
    print(f"  EN: {en_diff['b_hit_a_miss']}  (n={len(en_diff['b_hit_a_miss'])})")
    print(f"  ZH: {zh_diff['b_hit_a_miss']}  (n={len(zh_diff['b_hit_a_miss'])})")

    # -----------------------------------------------------------------------
    # 写报告
    # -----------------------------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "coarse_top_k": COARSE_TOP_K,
            "final_k": FINAL_K,
            "recall_at": RECALL_AT,
            "filtered": "answerable_only",  # 记录口径，避免与主评估报告混淆
        },
        "en": en_metrics,
        "zh": zh_metrics,
        "per_question": records,
        "diff": {"en": en_diff, "zh": zh_diff},
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print()
    print(f"报告已写入 {REPORT_PATH}")
    print(f"总耗时 {time.time() - t_start:.1f} 秒")


if __name__ == "__main__":
    main()