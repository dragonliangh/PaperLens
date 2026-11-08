"""第 7 块验收测试：评估逻辑与拒答逻辑。

运行：pytest test_eval.py -v
注意：需要 data/sample.pdf 已索引，否则 search() 返回空列表。

【历史修改痕迹】
  - 第 7 块（第一轮）：创建 7 个测试，覆盖 _hit / _first_hit_rank / 评估集结构 /
    拒答 / 正常回答。
  - 第 7 块（第二轮，本轮）：新增 test_thresholds_dict（验证字典格式）、
    test_get_threshold_fallback（验证 lang 缺失时回退到 default）。
"""
import json
import os

import pytest

from index_store import build_index, clear_index
from ingest import parse_and_chunk
from qa import REFUSAL_TEXT, answer
from evaluate import _first_hit_rank, _get_threshold, _hit, load_eval_set, load_thresholds

SAMPLE = "data/sample.pdf"
needs_sample = pytest.mark.skipif(not os.path.exists(SAMPLE), reason="缺少 sample.pdf")


@pytest.fixture(scope="module")
def indexed():
    clear_index()
    build_index(parse_and_chunk(SAMPLE))


def test_hit_interval():
    """_hit 使用区间判定：gold_page 落在 [page_start, page_end] 即命中。"""
    class FakeChunk:
        page_start = 3
        page_end = 5

    assert _hit(FakeChunk(), 4) is True
    assert _hit(FakeChunk(), 3) is True
    assert _hit(FakeChunk(), 5) is True
    assert _hit(FakeChunk(), 2) is False
    assert _hit(FakeChunk(), 6) is False


def test_first_hit_rank():
    """_first_hit_rank 返回 1-based 排名；未命中返回 None。"""
    class FakeChunk:
        def __init__(self, s, e):
            self.page_start = s
            self.page_end = e

    class FakeRC:
        def __init__(self, s, e):
            self.chunk = FakeChunk(s, e)

    results = [FakeRC(1, 1), FakeRC(4, 5), FakeRC(5, 6)]
    assert _first_hit_rank(results, 4) == 2
    assert _first_hit_rank(results, 6) == 3
    assert _first_hit_rank(results, 10) is None


def test_eval_set_size():
    """评估集应有 140 条，en 70 + zh 70。"""
    items = load_eval_set("eval_set.json")
    assert len(items) == 140
    assert sum(1 for it in items if it["lang"] == "en") == 70
    assert sum(1 for it in items if it["lang"] == "zh") == 70


def test_eval_set_id_continuity():
    """en 和 zh 的 id 必须严格 001–070 连续。"""
    items = load_eval_set("eval_set.json")
    en_ids = sorted(it["id"] for it in items if it["lang"] == "en")
    zh_ids = sorted(it["id"] for it in items if it["lang"] == "zh")
    assert en_ids == [f"en_{i:03d}" for i in range(1, 71)]
    assert zh_ids == [f"zh_{i:03d}" for i in range(1, 71)]


def test_eval_set_categories():
    """不可答题应有 8 unrelated + 8 semi_related + 4 counterfactual（每语言）。"""
    items = load_eval_set("eval_set.json")
    unans = [it for it in items if not it["answerable"]]
    cats = {}
    for it in unans:
        cats[it["category"]] = cats.get(it["category"], 0) + 1
    assert cats.get("unrelated") == 16
    assert cats.get("semi_related") == 16
    assert cats.get("counterfactual") == 8


def test_thresholds_dict():
    """thresholds.json 应为字典格式，含 default / en / zh。"""
    t = load_thresholds("thresholds.json")
    assert isinstance(t, dict)
    assert "default" in t
    assert "en" in t
    assert "zh" in t


def test_get_threshold_fallback():
    """_get_threshold：lang 缺失时回退到 default。"""
    t = {"default": 0.3, "en": 0.3, "zh": 0.18}
    assert _get_threshold(t, "en") == 0.3
    assert _get_threshold(t, "zh") == 0.18
    assert _get_threshold(t, "fr") == 0.3  # 未知语言回退


@needs_sample
def test_refusal_on_unrelated(indexed):
    """不可答题应被拒答。"""
    r = answer("What did the authors eat for lunch during the experiments?")
    assert r.answer == REFUSAL_TEXT
    assert r.citations == []


@needs_sample
def test_answer_on_answerable(indexed):
    """可答题应正常回答，不被误拒。"""
    r = answer("How much does LoRA reduce the number of trainable parameters?")
    assert r.answer != REFUSAL_TEXT
    assert len(r.citations) > 0


@needs_sample
def test_chinese_query_uses_zh_threshold(indexed):
    """中文 query 应走 zh 阈值（0.18），英文走 en 阈值（0.3）。

    验证方式：中文问「LoRA 将可训练参数数量减少了多少？」这个可答题，
    按旧阈值 0.3 会被拒，按新阈值 0.18 应正常回答。
    """
    r = answer("LoRA 将可训练参数数量减少了多少？")
    assert r.answer != REFUSAL_TEXT