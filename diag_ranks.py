"""评估明细分析器：把一次评估跑出来的逐题明细分段成表。

【为什么不是重新跑一遍】
  早期版本（临时诊断脚本）自己调 search + rerank 再跑 140 题，等于把
  evaluate.py 的 31 分钟再烧一遍，而且两次运行的分数不保证完全一致。
  现在改成读 `evaluate.py --details` 落盘的 `eval_details.json`，
  纯离线计算，同一批分数可以翻来覆去切片，不产生第二次推理开销。

【前置条件】
    python evaluate.py --details eval_details.json
  明细里每题带：id / lang / gold_page / coarse_rank / coarse_gold_score /
  coarse_top1_score / fine_rank / fine_gold_score / fine_top1_score / threshold / refused

【输出三张表 + 一张阈值敏感度表】
  表 1  排名分段：1 / 2 / 3 / 4 / 5 / 6-10 / 11-20 / miss，四分列（EN/ZH × 粗排/精排）
  表 2  分数分布：各分段内 top1 分数与命中分数（gold 分数）的 count/mean/P25/中位/P75
  表 3  区间迁移：粗排区间 -> 精排区间
  表 4  阈值敏感度：同一批分数在 0.15 / 0.18 / 0.30 下的拒答率与误拒率
        （这是 zh 0.15 vs 0.18 正式对比的数据底座）

【用法】
    python diag_ranks.py                       # 读 eval_details.json，打印三表
    python diag_ranks.py --out 表.md            # 同时落盘 Markdown
"""
import argparse
import json
import statistics

DETAILS_PATH = "eval_details.json"
BINS = ["1", "2", "3", "4", "5", "6-10", "11-20", "miss"]


def get_bin(rank):
    """把 1-based 排名映射到区间桶；None 记为 miss。"""
    if rank is None:
        return "miss"
    if rank <= 5:
        return str(rank)
    if rank <= 10:
        return "6-10"
    if rank <= 20:
        return "11-20"
    return ">20"


def _pct(part, total):
    return f"{part / total * 100:.1f}%" if total else "-"


def _quantile(values, q):
    """线性插值分位数（不依赖 numpy）。"""
    if not values:
        return None
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _fmt(v, nd=4):
    return "-" if v is None else f"{v:.{nd}f}"


def _score_stats(values):
    """一组分数 -> (n, mean, p25, median, p75)。"""
    if not values:
        return 0, None, None, None, None
    return (
        len(values),
        statistics.fmean(values),
        _quantile(values, 0.25),
        _quantile(values, 0.5),
        _quantile(values, 0.75),
    )


def build_tables(details):
    """把明细组织成四张表需要的数据结构（不打印，纯数据）。"""
    ans = [d for d in details if d["answerable"]]
    unans = [d for d in details if not d["answerable"]]

    seg = {}
    for lang in ("en", "zh"):
        la = [d for d in ans if d["lang"] == lang]
        seg[lang] = {
            "n": len(la),
            "coarse": {b: 0 for b in BINS},
            "fine": {b: 0 for b in BINS},
        }
        for d in la:
            seg[lang]["coarse"][get_bin(d["coarse_rank"])] += 1
            seg[lang]["fine"][get_bin(d["fine_rank"])] += 1

    # 分数分布：按「精排分段」分组，看该段内 top1 分数与 gold 分数
    dist = {}
    for lang in ("en", "zh"):
        la = [d for d in ans if d["lang"] == lang]
        dist[lang] = {}
        for b in BINS:
            bucket = [d for d in la if get_bin(d["fine_rank"]) == b]
            dist[lang][b] = {
                "top1": _score_stats([d["fine_top1_score"] for d in bucket if d["fine_top1_score"] is not None]),
                "gold": _score_stats([d["fine_gold_score"] for d in bucket if d["fine_gold_score"] is not None]),
            }

    # 区间迁移：粗排桶 -> 精排桶
    mig = {lang: {} for lang in ("en", "zh")}
    for lang in ("en", "zh"):
        for cb in BINS:
            row = {fb: 0 for fb in BINS}
            for d in ans:
                if d["lang"] != lang:
                    continue
                if get_bin(d["coarse_rank"]) != cb:
                    continue
                row[get_bin(d["fine_rank"])] += 1
            mig[lang][cb] = row

    # 阈值敏感度：同一批分数，不同拒答阈值
    thresh = {}
    for lang in ("en", "zh"):
        thresh[lang] = {
            "ans_top1": sorted(d["fine_top1_score"] for d in ans
                               if d["lang"] == lang and d["fine_top1_score"] is not None),
            "unans_top1": sorted(d["fine_top1_score"] for d in unans
                                 if d["lang"] == lang and d["fine_top1_score"] is not None),
        }
    return {"seg": seg, "dist": dist, "mig": mig, "thresh": thresh, "ans": ans, "unans": unans}


def render_markdown(t):
    """把数据结构渲染成 Markdown（表格形式，便于贴进 README / 报告）。"""
    out = []
    out.append("# 评估明细：排名分段 / 分数分布 / 阈值敏感度\n")
    out.append("数据来源：`evaluate.py --details` 单次评估（同一批分数离线切片）\n")

    # 表 1
    out.append("\n## 表 1　排名分段（命中 gold 的第一名次落在哪一段）\n")
    out.append("| 区间 | EN 粗排 | EN 精排 | ZH 粗排 | ZH 精排 |")
    out.append("| --- | --- | --- | --- | --- |")
    n_en, n_zh = t["seg"]["en"]["n"], t["seg"]["zh"]["n"]
    for b in BINS:
        out.append(
            f"| {b} | {t['seg']['en']['coarse'][b]} ({_pct(t['seg']['en']['coarse'][b], n_en)}) "
            f"| {t['seg']['en']['fine'][b]} ({_pct(t['seg']['en']['fine'][b], n_en)}) "
            f"| {t['seg']['zh']['coarse'][b]} ({_pct(t['seg']['zh']['coarse'][b], n_zh)}) "
            f"| {t['seg']['zh']['fine'][b]} ({_pct(t['seg']['zh']['fine'][b], n_zh)}) |"
        )
    out.append(f"\n可答题数：EN {n_en}，ZH {n_zh}。K=4 的口径 = 区间 1~4 之和；K=5 = 1~5 之和。\n")

    # 表 2
    out.append("\n## 表 2　分数分布（按精排分段）\n")
    out.append("`top1` = 该题精排第 1 名的 rerank 分数（决定拒答与否的那个数）；"
               "`gold` = 命中 chunk 自身的 rerank 分数。\n")
    for lang, name in (("en", "EN"), ("zh", "ZH")):
        out.append(f"\n### {name}\n")
        out.append("| 精排区间 | 题数 | top1 均值 | top1 P25 | top1 中位 | top1 P75 | gold 均值 | gold 中位 |")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for b in BINS:
            d = t["dist"][lang][b]
            tn, tmean, tp25, tmed, tp75 = d["top1"]
            gn, gmean, _gp25, gmed, _gp75 = d["gold"]
            out.append(f"| {b} | {tn} | {_fmt(tmean)} | {_fmt(tp25)} | {_fmt(tmed)} | {_fmt(tp75)} "
                       f"| {_fmt(gmean)} | {_fmt(gmed)} |")
        out.append("")

    # 表 3
    out.append("\n## 表 3　区间迁移（粗排 -> 精排）\n")
    for lang, name in (("en", "EN"), ("zh", "ZH")):
        out.append(f"\n### {name}\n")
        out.append("| 粗排原区间 | 精排 1 | 2 | 3 | 4 | 5 | 6-10 | 11-20 | miss |")
        out.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for cb in BINS:
            if sum(t["mig"][lang][cb].values()) == 0:
                continue
            cells = " | ".join(str(t["mig"][lang][cb][fb]) for fb in BINS)
            out.append(f"| {cb} | {cells} |")
        out.append("")

    # 表 4
    out.append("\n## 表 4　阈值敏感度（同一批分数）\n")
    out.append("| 语言 | 阈值 | 可答题误拒率 | 不可答题拒答率 | 可答 top1 中位 | 不可答 top1 中位 |")
    out.append("| --- | --- | --- | --- | --- | --- |")
    for lang, name in (("en", "EN"), ("zh", "ZH")):
        ta = t["thresh"][lang]["ans_top1"]
        tu = t["thresh"][lang]["unans_top1"]
        for th in (0.15, 0.18, 0.30):
            fr = sum(1 for v in ta if v < th)
            rr = sum(1 for v in tu if v < th)
            out.append(
                f"| {name} | {th:.2f} | {_pct(fr, len(ta))} ({fr}/{len(ta)}) "
                f"| {_pct(rr, len(tu))} ({rr}/{len(tu)}) "
                f"| {_fmt(_quantile(ta, 0.5))} | {_fmt(_quantile(tu, 0.5))} |"
            )
    out.append("\n注：误拒率 = 可答题 top1 分数低于阈值（会被误拒）；拒答率 = 不可答题 top1 分数低于阈值（正确拒答）。\n")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="评估逐题明细 -> 分段表 / 分数分布 / 阈值敏感度")
    parser.add_argument("--details", default=DETAILS_PATH, help=f"明细 JSON（默认 {DETAILS_PATH}）")
    parser.add_argument("--out", default=None, help="Markdown 输出路径；不给只打印")
    args = parser.parse_args()

    with open(args.details, encoding="utf-8") as f:
        details = json.load(f)
    if not details:
        raise SystemExit(f"{args.details} 是空的：先跑 evaluate.py --details")

    tables = build_tables(details)
    md = render_markdown(tables)
    print(md)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n已写入 {args.out}")


if __name__ == "__main__":
    main()
