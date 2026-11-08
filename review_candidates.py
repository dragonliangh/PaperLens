"""人工核验脚本：对争议题打印粗排 top-20 的原文，供人工判断。

【目的】
  上一轮发现 12 道题「粗排命中但精排丢」，但部分题的 gold 标注可能有争议。
  本脚本打印这 12 道题的粗排 top-20 每个 chunk 的：
    - 排名、页码、余弦分数
    - 前 150 字符原文
  人工逐题阅读后判断：
    (a) gold_page 标注正确，精排确实丢了答案 → 记录为「精排弱点」
    (b) gold_page 标注错误，正确答案在其他页 → 修正 gold_page
    (c) gold_page 不唯一，多个页面都是直接答案 → 记录为「题目语义模糊」，考虑删题

【为什么只跑 12 道】
  全量 50 道可答题 × 20 个 chunk = 1000 段原文，人工看不完。
  只跑有争议的 12 道，约 240 段原文，1-2 小时能看完。

【输出】
  打印到终端 + 同时写入 review_candidates.txt，方便慢慢看。
"""
from index_store import search
from evaluate import load_eval_set

EVAL_SET_PATH = "eval_set.json"
COARSE_TOP_K = 20
OUTPUT_FILE = "review_candidates.txt"

# 争议题列表：上一轮诊断中「粗排命中但精排丢」的 12 道。
TARGET_IDS = [
    "en_003", "en_006", "en_013", "en_014",
    "en_016", "en_029", "en_038", "en_040",
    "en_042", "en_044", "en_045", "en_048",
]


def _get_gold_desc(item: dict) -> str:
    """返回 gold 的描述（兼容 gold_page 和 gold_pages 两种格式）。"""
    if "gold_pages" in item and item["gold_pages"]:
        return f"gold_pages = {item['gold_pages']}"
    return f"gold_page = {item.get('gold_page')}"


def _is_gold(item: dict, chunk) -> bool:
    """判断 chunk 是否命中 gold。"""
    pages = item.get("gold_pages")
    if pages:
        return any(chunk.page_start <= p <= chunk.page_end for p in pages)
    gp = item.get("gold_page")
    if gp is None:
        return False
    return chunk.page_start <= gp <= chunk.page_end


def review_candidates():
    items = load_eval_set(EVAL_SET_PATH)
    item_map = {it["id"]: it for it in items if it["id"] in TARGET_IDS}

    missing = set(TARGET_IDS) - set(item_map.keys())
    if missing:
        raise ValueError(f"评估集中找不到这些 id：{missing}")

    lines = []

    def emit(s: str = ""):
        """同时打印到终端和写入文件。"""
        print(s)
        lines.append(s)

    for tid in TARGET_IDS:
        it = item_map[tid]
        q = it["question"]

        emit("=" * 80)
        emit(f"{tid}: {q}")
        emit(f"{_get_gold_desc(it)}")
        emit(f"gold_keywords = {it.get('gold_keywords', [])}")
        emit("=" * 80)

        coarse = search(q, top_k=COARSE_TOP_K)
        for i, r in enumerate(coarse, start=1):
            marker = " ★ GOLD" if _is_gold(it, r.chunk) else ""
            page = f"p{r.chunk.page_start}-{r.chunk.page_end}"
            snippet = r.chunk.text[:150].replace("\n", " ")
            emit(f"  #{i:<2} {page:<10} score={r.score:.4f}{marker}")
            emit(f"       {snippet!r}")

        emit()

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n完整输出已写入 {OUTPUT_FILE}，可用 VS Code 打开慢慢看。")


if __name__ == "__main__":
    review_candidates()