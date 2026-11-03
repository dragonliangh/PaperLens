"""第 1 块验收测试：对应契约 8a-8f（详见 DESIGN.md）。

运行方式（在项目根目录、已激活 .venv 的终端里）：
    pytest test_ingest.py -v
"""
import os
# 强制离线，避免测试时 transformers 尝试联网做版本检查而卡住。
# 这两行必须放在 import transformers 相关模块之前才生效。
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest

from ingest import parse_and_chunk

SAMPLE = "data/sample.pdf"

# 如果样例 PDF 不存在，就跳过依赖它的 3 个测试，而不是报错。
# 这样别人 clone 仓库、还没放 PDF 时，跑 pytest 也不会一片红。
needs_sample = pytest.mark.skipif(
    not os.path.exists(SAMPLE), reason="缺少 data/sample.pdf，先放一篇论文再跑"
)


@needs_sample
def test_8a_non_empty():
    """契约 8a：真实论文 PDF 解析后应返回非空 chunk 列表。"""
    chunks = parse_and_chunk(SAMPLE)
    assert len(chunks) > 0


@needs_sample
def test_8b_fields():
    """契约 8b：每个 chunk 的字段必须合法。

    检查四项：
      - text 非空白（不能是空块）；
      - doc_name 等于文件名（引用要靠它显示来源）；
      - page_start >= 1（页码从 1 开始，不能是 0 或负数）；
      - page_end >= page_start（结束页不能早于起始页）。
    """
    chunks = parse_and_chunk(SAMPLE)
    for c in chunks:
        assert c.text.strip()
        assert c.doc_name == "sample.pdf"
        assert c.page_start >= 1
        assert c.page_end >= c.page_start


@needs_sample
def test_8c_chunk_id_unique():
    """契约 8c：chunk_id 必须全局唯一。

    做法：把全部 id 放进集合，比较「列表长度」和「集合长度」是否相等；
    相等说明没有重复。
    """
    chunks = parse_and_chunk(SAMPLE)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_8e_missing_file():
    """契约 8e：文件不存在时抛 FileNotFoundError（不吞异常）。"""
    with pytest.raises(FileNotFoundError):
        parse_and_chunk("data/does_not_exist.pdf")


def test_8f_not_a_pdf(tmp_path):
    """契约 8f：文件存在但不是合法 PDF 时，解析异常要向上抛（不吞）。

    tmp_path 是 pytest 提供的临时目录 fixture，测试结束自动清理。
    这里造一个后缀是 .pdf、内容却是纯文本的假文件。
    不限定具体异常类型（不同 PDF 库抛的不同），只要是异常即可。
    """
    fake = tmp_path / "fake.pdf"
    fake.write_bytes(b"this is not a pdf")
    with pytest.raises(Exception):
        parse_and_chunk(str(fake))


# 契约 8d（覆盖率）为人工抽查，不写自动断言：
# 打开 chunk 列表，随机取 2-3 个 chunk 对照原 PDF 页码核对文字，
# 确认没有大面积丢字、页码对得上。第 1 块验收时已人工执行并通过。