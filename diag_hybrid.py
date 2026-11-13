"""临时诊断：查明混合检索为何无净提升（D1-D5）。跑完可删。

【用途】
  D1 融合是否生效（含 bm25_only_count 三步判断）
  D2 4 道变化题明细（query + A/B 粗排 top5 + BM25 独有 chunk）
  D3 BM25 单独 R@5/R@20 + 与向量重合率（对照随机基线）
  D4 题型统计（专名型 / 数值型 / 概括型 分栏）
  D5 McNemar 精确检验 + bootstrap R@5 差值 95% CI

【遵循】
  只读，不改任何生产代码；不调 LLM；不跑重排。
"""
import json
import random
from math import comb

from index_store import _get_collection, search as vector_search
from hybrid_search import (
    bm25_search, search_hybrid, _tokenize, _get_state,
)
from evaluate import load_eval_set, _first_hit_rank

HYBRID_REPORT = "hybrid_eval_report.json"
EVAL_SET = "eval_set.json"
OUT_REPORT = "diag_hybrid_report.json"

CHANGE_IDS = ["en_019", "en_045", "en_005", "en_034"]
COARSE_TOP_K = 20


def d1_fusion_effective(items, hyb_report):
    """D1：融合是否生效（三步判断）。

    对每题算 bm25_only = bm25_top20 - vector_top20；
    再看 B 组（hybrid）里是否含这些 bm25_only 的 chunk。
    """
    answerable = [it for it in items if it["answerable"]]

    total_bm25_only = 0
    fusion_effective_count = 0
    fusion_swallowed_count = 0
    same_route_count = 0

    details = []
    for it in answerable:
        q = it["question"]
        vec = vector_search(q, top_k=COARSE_TOP_K)
        bm = bm25_search(q, top_k=COARSE_TOP_K)
        hyb = search_hybrid(q, top_k=COARSE_TOP_K)

        # 【修正 A】RetrievedChunk 的 chunk_id 在 .chunk 上；Chunk 本身有 chunk_id。
        vec_ids = {r.chunk.chunk_id for r in vec}
        bm_ids = {c.chunk_id for c in bm}
        hyb_ids = {r.chunk.chunk_id for r in hyb}
        bm_only = bm_ids - vec_ids

        total_bm25_only += len(bm_only)

        if len(bm_only) == 0:
            same_route_count += 1
        elif len(bm_only & hyb_ids) > 0:
            fusion_effective_count += 1
        else:
            fusion_swallowed_count += 1
            details.append({"id": it["id"], "bm25_only_n": len(bm_only),
                            "hyb_has_bm25_only": False})

    if total_bm25_only == 0:
        verdict = "两路完全同质（bm25_only_count=0），B=A 是正常的，非 bug。"
    elif fusion_swallowed_count == 0:
        verdict = "融合生效：B 组包含了 BM25 独有候选。"
    else:
        verdict = (f"融合可能被吞：{fusion_swallowed_count} 题有 BM25 独有候选"
                   f"但 B 组未含。")

    return {
        "total_bm25_only_chunks": total_bm25_only,
        "same_route_questions": same_route_count,
        "fusion_effective_questions": fusion_effective_count,
        "fusion_swallowed_questions": fusion_swallowed_count,
        "verdict": verdict,
        "swallowed_details": details[:10],
    }


def d2_changed_questions(items):
    """D2：4 道变化题明细。"""
    item_map = {it["id"]: it for it in items if it["id"] in CHANGE_IDS}
    out = []
    for tid in CHANGE_IDS:
        it = item_map[tid]
        q = it["question"]
        gold = it["gold_page"]
        vec = vector_search(q, top_k=5)
        bm = bm25_search(q, top_k=5)
        hyb = search_hybrid(q, top_k=5)

        def fmt_from_retrieved(results):
            # 【修正 A】RetrievedChunk -> .chunk.chunk_id / .chunk.page_start
            return [
                {"rank": i, "chunk_id": r.chunk.chunk_id,
                 "pages": f"p{r.chunk.page_start}-{r.chunk.page_end}",
                 "is_gold": r.chunk.page_start <= gold <= r.chunk.page_end,
                 "snippet": r.chunk.text[:80]}
                for i, r in enumerate(results, 1)
            ]

        def fmt_from_chunks(chunks):
            # BM25 路返回 list[Chunk]，直接取字段
            return [
                {"rank": i, "chunk_id": c.chunk_id,
                 "pages": f"p{c.page_start}-{c.page_end}",
                 "is_gold": c.page_start <= gold <= c.page_end,
                 "snippet": c.text[:80]}
                for i, c in enumerate(chunks, 1)
            ]

        out.append({
            "id": tid, "query": q, "gold_page": gold,
            "vector_top5": fmt_from_retrieved(vec),
            "bm25_top5": fmt_from_chunks(bm),
            "hybrid_top5": fmt_from_retrieved(hyb),
        })
    return out


def d3_bm25_standalone(items):
    """D3：BM25 单独 R@5 / R@20 + 与向量重合率（对照随机基线）。"""
    answerable = [it for it in items if it["answerable"]]
    n = len(answerable)

    corpus_size = _get_collection().count()
    random_r5 = min(5, corpus_size) / corpus_size if corpus_size else 0
    random_r20 = min(20, corpus_size) / corpus_size if corpus_size else 0
    random_overlap_20 = (20 * 20 / corpus_size) if corpus_size else 0

    # 【修正】_first_hit_rank 期待 RetrievedChunk（有 .chunk），BM25 返回 Chunk。
    # 包一层 RetrievedChunk，score 填 0（命中判定只用 .chunk，不用 score）。
    from index_store import RetrievedChunk

    def _wrap(chunks):
        return [RetrievedChunk(chunk=c, score=0.0) for c in chunks]

    bm25_hit5 = 0
    bm25_hit20 = 0
    overlap_sum = 0
    overlap_n = 0

    for it in answerable:
        q = it["question"]
        gold = it["gold_page"]
        bm = bm25_search(q, top_k=COARSE_TOP_K)
        wrapped = _wrap(bm)
        if _first_hit_rank(wrapped[:5], gold) is not None:
            bm25_hit5 += 1
        if _first_hit_rank(wrapped, gold) is not None:
            bm25_hit20 += 1

        vec = vector_search(q, top_k=COARSE_TOP_K)
        vec_ids = {r.chunk.chunk_id for r in vec}
        bm_ids = {c.chunk_id for c in bm}
        if bm_ids:
            overlap_sum += len(bm_ids & vec_ids)
            overlap_n += 1

    return {
        "corpus_size": corpus_size,
        "n_questions": n,
        "bm25_r5": round(bm25_hit5 / n, 4) if n else 0,
        "bm25_r20": round(bm25_hit20 / n, 4) if n else 0,
        "random_r5_baseline": round(random_r5, 4),
        "random_r20_baseline": round(random_r20, 4),
        "bm25_r20_vs_random": round((bm25_hit20 / n) / random_r20, 2) if n and random_r20 else 0,
        "avg_overlap_20": round(overlap_sum / overlap_n, 2) if overlap_n else 0,
        "random_overlap_20": round(random_overlap_20, 2),
        "overlap_rate": round((overlap_sum / overlap_n) / 20, 4) if overlap_n else 0,
    }


def d4_question_types(items):
    """D4：题型统计（专名型 / 数值型 / 概括型 分栏）。"""
    answerable = [it for it in items if it["answerable"]]

    proper = 0
    numeric = 0
    proper_or_numeric = 0
    generic = 0

    for it in answerable:
        kws = " ".join(it.get("gold_keywords") or [])
        has_upper = any(ch.isupper() and ch.isascii() for ch in kws)
        has_digit = any(ch.isdigit() for ch in kws)
        if has_upper:
            proper += 1
        if has_digit:
            numeric += 1
        if has_upper or has_digit:
            proper_or_numeric += 1
        else:
            generic += 1

    n = len(answerable)
    return {
        "n": n,
        "proper_noun": proper,
        "numeric": numeric,
        "proper_or_numeric": proper_or_numeric,
        "generic": generic,
        "proper_rate": round(proper / n, 4) if n else 0,
        "numeric_rate": round(numeric / n, 4) if n else 0,
        "generic_rate": round(generic / n, 4) if n else 0,
    }


def _mcnemar_exact(b, c):
    """McNemar 精确二项双尾 p。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def _bootstrap_ci(diffs, n_boot=2000, alpha=0.05, seed=42):
    """bootstrap 差值 95% CI。"""
    rnd = random.Random(seed)
    n = len(diffs)
    if n == 0:
        return (0.0, 0.0)
    boot = []
    for _ in range(n_boot):
        sample = [rnd.choice(diffs) for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    lo = boot[int((alpha / 2) * n_boot)]
    hi = boot[int((1 - alpha / 2) * n_boot) - 1]
    return (round(lo, 4), round(hi, 4))


def d5_mcnemar(items, hyb_report):
    """D5：McNemar 精确检验 + bootstrap R@5 差值 95% CI。"""
    per_q = {r["id"]: r for r in hyb_report["per_question"]}
    answerable = [it for it in items if it["answerable"]]

    # 【修正 B】四个计数器统一命名 a / b_ / c_ / d
    a = b_ = c_ = d = 0
    r5_diffs = []
    for it in answerable:
        r = per_q.get(it["id"])
        if r is None:
            continue
        av = r["vector_only"]["fine_rank"] is not None
        bv = r["hybrid"]["fine_rank"] is not None
        if av and bv:
            a += 1
        elif av and not bv:
            b_ += 1
        elif not av and bv:
            c_ += 1
        else:
            d += 1

        av5 = (r["vector_only"]["coarse_rank"] is not None
               and r["vector_only"]["coarse_rank"] <= 5)
        bv5 = (r["hybrid"]["coarse_rank"] is not None
               and r["hybrid"]["coarse_rank"] <= 5)
        r5_diffs.append((1 if bv5 else 0) - (1 if av5 else 0))

    n_disc = b_ + c_
    if n_disc < 6:
        p = None
        note = f"不一致对仅 {n_disc} 个（b={b_}, c={c_}），样本量不足以做检验"
    else:
        p = _mcnemar_exact(b_, c_)
        note = f"b={b_}（A赢）, c={c_}（B赢）, n={n_disc}, p={p:.4f}"

    ci = _bootstrap_ci(r5_diffs)
    return {
        "table": {"a_both_hit": a, "b_A_hit_B_miss": b_,
                  "c_A_miss_B_hit": c_, "d_both_miss": d},
        "mcnemar_n": n_disc,
        "mcnemar_p": p,
        "note": note,
        "r5_diff_mean": round(sum(r5_diffs) / len(r5_diffs), 4) if r5_diffs else 0,
        "r5_diff_ci95": ci,
        "ci_contains_zero": ci[0] <= 0 <= ci[1],
    }


def main():
    items = load_eval_set(EVAL_SET)
    with open(HYBRID_REPORT, "r", encoding="utf-8") as f:
        hyb_report = json.load(f)

    print("=" * 70)
    print("D1：融合是否生效")
    print("=" * 70)
    d1 = d1_fusion_effective(items, hyb_report)
    for k, v in d1.items():
        if k != "swallowed_details":
            print(f"  {k}: {v}")
    print()

    print("=" * 70)
    print("D2：4 道变化题明细")
    print("=" * 70)
    d2 = d2_changed_questions(items)
    for rec in d2:
        print(f"\n  {rec['id']}: {rec['query']}")
        print(f"    gold_page={rec['gold_page']}")
        for grp in ["vector_top5", "bm25_top5", "hybrid_top5"]:
            print(f"    [{grp}]")
            for r in rec[grp]:
                mark = " *" if r["is_gold"] else ""
                print(f"      #{r['rank']} {r['pages']} {r['chunk_id']}{mark}")
    print()

    print("=" * 70)
    print("D3：BM25 单独表现 + 与向量重合率")
    print("=" * 70)
    d3 = d3_bm25_standalone(items)
    for k, v in d3.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 70)
    print("D4：题型统计")
    print("=" * 70)
    d4 = d4_question_types(items)
    for k, v in d4.items():
        print(f"  {k}: {v}")
    print()

    print("=" * 70)
    print("D5：McNemar + bootstrap CI")
    print("=" * 70)
    d5 = d5_mcnemar(items, hyb_report)
    for k, v in d5.items():
        print(f"  {k}: {v}")
    print()

    report = {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "D5": d5}
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已写入 {OUT_REPORT}")


if __name__ == "__main__":
    main()