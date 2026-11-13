"""PaperLens 阶段 2b：混合检索评估。

【模块职责】
  量化「纯向量检索」vs「混合检索（BM25+向量+RRF）」的指标差异，
  产出可直接写进 README 的 A/B 数字与每题明细。

【在架构中的位置】
  阶段 2 的独立评估脚本，不属于生产链路。
    - 上游依赖：index_store.search（A 组纯向量）、hybrid_search.search_hybrid（B 组混合）、
      rerank.rerank（精排）、evaluate 的 load_eval_set / _first_hit_rank / _hit（复用口径）。
    - 下游被依赖：无。README「混合检索」一节的数字来源。
    - 明确不修改：evaluate.py / qa.py / index_store.py / rerank.py / hybrid_search.py。

【类级】
  本文件无类定义，全部为模块级函数（纯函数脚本，与 evaluate.py / evaluate_ablation.py 同风格）。

【对外暴露】
  - main() -> None ：跑完整 A/B 评估，打印表格 + 写 hybrid_eval_report.json。

【遵循的契约（阶段 2b）】
  - 2b.1：量化混合检索 vs 纯向量的指标差异。
  - 2b.4 A 组：index_store.search(top_k=20) → 算粗排指标 → rerank(final_k=4) → 算精排指标。
  - 2b.5 B 组：hybrid_search.search_hybrid(top_k=20) → 同上。
  - 2b.6：粗排 Recall@5 / MRR@20；精排 Recall@4 / MRR@4；中英文分开。
  - 2b.7：空索引照常跑完；rerank 异常向上抛；不打日志。
  - E2：只算可答题（100 道）。
  - E5 / 20e：中文 A/B 应逐题完全一致（透传验证），不一致 → 打印题号 + raise（缺口 c）。

【遵循的设计决策】
  - E1：A 组显式 from index_store import search as vector_search，
        避免 qa.py 的 search_hybrid 别名污染 A 组。
  - E2：每题跑 A/B 两组，各 rerank 一次（候选池不同，不能共用）。
  - E3：每题存 coarse_rank + fine_rank 两个整数（同时支撑 Recall 与 MRR）。
  - E4：diff 算混合救回 / 推走，与阶段 1 消融 diff 结构对称。
  - E6：不写 test_evaluate_hybrid.py，验收靠跑脚本看报告。
  - E7：rerank 异常向上抛。

【依赖】
  标准库 json / time / datetime；
  index_store.search（A 组）；hybrid_search.search_hybrid（B 组）；
  rerank.rerank；evaluate 的 load_eval_set / _first_hit_rank。
"""
import json
import time
from datetime import datetime

from index_store import search as vector_search   # E1：A 组纯向量，显式别名
from hybrid_search import search_hybrid           # B 组混合
from rerank import rerank
from evaluate import load_eval_set, _first_hit_rank

# 文件路径：集中定义，便于测试时替换。
EVAL_SET_PATH = "eval_set.json"
REPORT_PATH = "hybrid_eval_report.json"

# 口径参数：与 evaluate.py / evaluate_ablation.py 保持一致，便于交叉核对。
COARSE_TOP_K = 20
FINAL_K = 4
RECALL_AT_5 = 5
RECALL_AT_4 = 4
PROGRESS_EVERY = 10


def _aggregate(records: list[dict], group: str) -> dict:
    """把一组题（同语言）某组（vector_only / hybrid）的每题记录聚合成指标。

    【方法契约】
      输入：records = [{id, lang, gold_page,
                        vector_only:{coarse_rank, fine_rank},
                        hybrid:{coarse_rank, fine_rank}}, ...]
            group = "vector_only" | "hybrid"
      输出：{recall_at_5, mrr_at_20, recall_at_4, mrr_at_4}

    【指标口径】
      Recall@5 = coarse_rank 在 1..5 的题数 / 总题数
      MRR@20   = sum(1/coarse_rank if coarse_rank else 0) / 总题数
      Recall@4 = fine_rank 在 1..4 的题数 / 总题数
      MRR@4    = sum(1/fine_rank if fine_rank else 0) / 总题数
    """
    n = len(records)
    if n == 0:
        # 【边界】空组防御：避免除零。契约 2b.7 只承诺空索引不崩，此处是额外兜底。
        return {"recall_at_5": 0.0, "mrr_at_20": 0.0,
                "recall_at_4": 0.0, "mrr_at_4": 0.0}

    coarse_hits_5 = sum(
        1 for r in records
        if r[group]["coarse_rank"] is not None
        and r[group]["coarse_rank"] <= RECALL_AT_5
    )
    coarse_mrr_20 = sum(
        1.0 / r[group]["coarse_rank"] if r[group]["coarse_rank"] is not None else 0.0
        for r in records
    ) / n
    fine_hits_4 = sum(
        1 for r in records
        if r[group]["fine_rank"] is not None
        and r[group]["fine_rank"] <= RECALL_AT_4
    )
    fine_mrr_4 = sum(
        1.0 / r[group]["fine_rank"] if r[group]["fine_rank"] is not None else 0.0
        for r in records
    ) / n

    return {
        "recall_at_5": round(coarse_hits_5 / n, 4),
        "mrr_at_20": round(coarse_mrr_20, 4),
        "recall_at_4": round(fine_hits_4 / n, 4),
        "mrr_at_4": round(fine_mrr_4, 4),
    }


def _diff(records: list[dict]) -> dict:
    """算「混合检索救回」（A 未命中 B 命中）与「混合推走」（A 命中 B 未命中）。

    口径用精排 Recall@4（fine_rank 非 None 即命中），与 README 叙事一致。
    """
    rescued, lost = [], []
    for r in records:
        a_hit = r["vector_only"]["fine_rank"] is not None
        b_hit = r["hybrid"]["fine_rank"] is not None
        if b_hit and not a_hit:
            rescued.append(r["id"])
        elif a_hit and not b_hit:
            lost.append(r["id"])
    return {"hybrid_rescued": rescued, "hybrid_lost": lost}


def _assert_zh_consistency(zh_records: list[dict]) -> None:
    """契约 20e：中文 A/B 应逐题完全一致（search_hybrid 对 zh 透传 search）。

    不一致 → 打印不一致题号 + raise AssertionError（缺口 c：崩溃前给证据）。
    """
    inconsistent = []
    for r in zh_records:
        a = r["vector_only"]
        b = r["hybrid"]
        if a["coarse_rank"] != b["coarse_rank"] or a["fine_rank"] != b["fine_rank"]:
            inconsistent.append(
                f"{r['id']} (vector coarse={a['coarse_rank']}, fine={a['fine_rank']}; "
                f"hybrid coarse={b['coarse_rank']}, fine={b['fine_rank']})"
            )
    if inconsistent:
        print("【契约 20e 失败】中文 A/B 不一致的题：")
        for line in inconsistent:
            print(f"  {line}")
        raise AssertionError(
            f"中文 A/B 应逐题一致，但有 {len(inconsistent)} 题不一致。"
            "根因通常是 search_hybrid 的中文透传分支有 bug。"
        )


def main() -> None:
    """跑完整 A/B 评估，打印表格 + 写 hybrid_eval_report.json。"""
    items = load_eval_set(EVAL_SET_PATH)
    # 【核心逻辑 / E2】只算可答题，与阶段 1 消融口径一致。
    answerable = [it for it in items if it["answerable"]]
    total = len(answerable)

    print(f"混合检索评估集：{total} 道可答题（原 {len(items)} 条，已过滤不可答题）")
    print(f"口径：A=纯向量 search(top_k={COARSE_TOP_K})→rerank(final_k={FINAL_K})；"
          f"B=混合 search_hybrid(top_k={COARSE_TOP_K})→rerank(final_k={FINAL_K})")
    print()

    records: list[dict] = []
    t_start = time.time()

    for i, it in enumerate(answerable, start=1):
        q = it["question"]
        gold_page = it["gold_page"]

        # A 组：纯向量（E1：显式 vector_search，不走 qa 的 hybrid 别名）
        vec_coarse = vector_search(q, top_k=COARSE_TOP_K)
        vec_coarse_rank = _first_hit_rank(vec_coarse, gold_page)
        # 【E7】rerank 异常向上抛，不吞。
        vec_fine = rerank(q, vec_coarse, final_k=FINAL_K)
        vec_fine_rank = _first_hit_rank(vec_fine, gold_page)

        # B 组：混合（E2：候选池不同，独立 rerank）
        hyb_coarse = search_hybrid(q, top_k=COARSE_TOP_K)
        hyb_coarse_rank = _first_hit_rank(hyb_coarse, gold_page)
        hyb_fine = rerank(q, hyb_coarse, final_k=FINAL_K)
        hyb_fine_rank = _first_hit_rank(hyb_fine, gold_page)

        records.append({
            "id": it["id"],
            "lang": it["lang"],
            "gold_page": gold_page,
            "vector_only": {"coarse_rank": vec_coarse_rank, "fine_rank": vec_fine_rank},
            "hybrid": {"coarse_rank": hyb_coarse_rank, "fine_rank": hyb_fine_rank},
        })

        if i % PROGRESS_EVERY == 0 or i == total:
            print(f"  进度 {i}/{total}，已用 {time.time() - t_start:.1f} 秒")

    en_records = [r for r in records if r["lang"] == "en"]
    zh_records = [r for r in records if r["lang"] == "zh"]

    # 【契约 20e / 缺口 c】中文 A/B 必须逐题一致，不一致直接崩溃（不写报告）。
    _assert_zh_consistency(zh_records)

    en_vec = _aggregate(en_records, "vector_only")
    en_hyb = _aggregate(en_records, "hybrid")
    zh_vec = _aggregate(zh_records, "vector_only")
    zh_hyb = _aggregate(zh_records, "hybrid")

    # -----------------------------------------------------------------------
    # 打印 A/B 对比表
    # -----------------------------------------------------------------------
    print()
    print(f"{'指标':<34} {'EN':>10} {'ZH':>10}")
    print("-" * 58)

    def _row_pair(label: str, en_a, en_b, zh_a, zh_b) -> None:
        # 【语句级】每个指标打两行：A 值 / B 值，便于肉眼对比。
        print(f"{label:<34} {en_a:>10.4f} {zh_a:>10.4f}  ← A(纯向量)")
        print(f"{'':<34} {en_b:>10.4f} {zh_b:>10.4f}  ← B(混合)")

    _row_pair("粗排 Recall@5",
              en_vec["recall_at_5"], en_hyb["recall_at_5"],
              zh_vec["recall_at_5"], zh_hyb["recall_at_5"])
    _row_pair("粗排 MRR@20",
              en_vec["mrr_at_20"], en_hyb["mrr_at_20"],
              zh_vec["mrr_at_20"], zh_hyb["mrr_at_20"])
    _row_pair("精排 Recall@4",
              en_vec["recall_at_4"], en_hyb["recall_at_4"],
              zh_vec["recall_at_4"], zh_hyb["recall_at_4"])
    _row_pair("精排 MRR@4",
              en_vec["mrr_at_4"], en_hyb["mrr_at_4"],
              zh_vec["mrr_at_4"], zh_hyb["mrr_at_4"])

    print()
    print("Δ（B - A）：")
    print(f"  粗排 Recall@5  EN {en_hyb['recall_at_5'] - en_vec['recall_at_5']:+.4f}  "
          f"ZH {zh_hyb['recall_at_5'] - zh_vec['recall_at_5']:+.4f}")
    print(f"  精排 Recall@4  EN {en_hyb['recall_at_4'] - en_vec['recall_at_4']:+.4f}  "
          f"ZH {zh_hyb['recall_at_4'] - zh_vec['recall_at_4']:+.4f}")
    print(f"  精排 MRR@4     EN {en_hyb['mrr_at_4'] - en_vec['mrr_at_4']:+.4f}  "
          f"ZH {zh_hyb['mrr_at_4'] - zh_vec['mrr_at_4']:+.4f}")

    # -----------------------------------------------------------------------
    # diff 明细
    # -----------------------------------------------------------------------
    en_diff = _diff(en_records)
    zh_diff = _diff(zh_records)
    print()
    print("混合救回（A 未命中 B 命中）：")
    print(f"  EN: {en_diff['hybrid_rescued']}  (n={len(en_diff['hybrid_rescued'])})")
    print(f"  ZH: {zh_diff['hybrid_rescued']}  (n={len(zh_diff['hybrid_rescued'])})")
    print("混合推走（A 命中 B 未命中）：")
    print(f"  EN: {en_diff['hybrid_lost']}  (n={len(en_diff['hybrid_lost'])})")
    print(f"  ZH: {zh_diff['hybrid_lost']}  (n={len(zh_diff['hybrid_lost'])})")

    # -----------------------------------------------------------------------
    # 写报告
    # -----------------------------------------------------------------------
    report = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "coarse_top_k": COARSE_TOP_K,
            "final_k": FINAL_K,
            "recall_at_5": RECALL_AT_5,
            "recall_at_4": RECALL_AT_4,
            "filtered": "answerable_only",
        },
        "en": {"answerable_count": len(en_records),
               "vector_only": en_vec, "hybrid": en_hyb},
        "zh": {"answerable_count": len(zh_records),
               "vector_only": zh_vec, "hybrid": zh_hyb},
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