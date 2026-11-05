"""PaperLens 第 1 块：PDF 解析 + 分块。

这个模块的职责很单一：把一篇 PDF 论文变成一串「文本块（chunk）」，
每个块自带出处信息（文档名、页码、章节），供后续第 2 块做向量索引。

对外只暴露两样东西：
  - Chunk：一个数据类，描述单个文本块。
  - parse_and_chunk()：唯一对外函数，输入 PDF 路径，输出 Chunk 列表。
"""
import os
# 强制离线：必须在 import transformers 之前设置才生效。
# 【关键补丁】PyTorch 2.6+ 默认 weights_only=True，会拒绝加载 .bin 权重中的非张量对象。
# 必须在 import transformers 之前 monkey-patch torch.load，强制 weights_only=False。
import torch as _torch
_orig_torch_load = _torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
_torch.load = _patched_torch_load

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import re                       # 用来写章节标题的正则
from dataclasses import dataclass   # 用 dataclass 定义 Chunk，字段有名字、可读性好
from functools import lru_cache     # 缓存 tokenizer，避免重复加载

import pdfplumber               # PDF 解析库
from transformers import AutoTokenizer  # 从 Hugging Face 加载分词器

# tokenizer 的本地路径。
# 为什么不是 "BAAI/bge-small-en-v1.5" 这个 HF 仓库名？
#   因为当前网络无法稳定访问 HF / hf-mirror，改为提前把文件下载到本地，
#   从本地目录加载，彻底离线。具体下载过程见 README「离线模型」一节。
TOKENIZER_NAME = "./models/bge-small-en-v1.5"


@dataclass
class Chunk:
    """一个文本块，是检索的最小单位。

    字段说明：
      text        : 块的正文（已按 token 窗口切好的原文片段）。
      doc_name    : 文档文件名，例如 "sample.pdf"。用于引用展示和 chunk_id。
      page_start  : 该块起始页码，从 1 开始。
      page_end    : 该块结束页码；跨页时 page_end > page_start。
      section     : 块内出现的第一个章节标题（如 "1 INTRODUCTION"）；
                    识别不到填空字符串。注意：它只表示「块里出现过这个标题」，
                    不保证整块都属于这一节（详见 DESIGN.md 决策 5）。
      chunk_id    : 全局唯一 ID，格式 "{doc_name}#p{起页}-{止页}#c{序号}"。
                    序号在单篇文档内递增，保证同一文档内唯一；
                    不同文档靠 doc_name 区分。用于评估标注、日志追溯、去重。
    """
    text: str
    doc_name: str
    page_start: int
    page_end: int
    section: str
    chunk_id: str


@lru_cache(maxsize=1)
def _get_tokenizer():
    """加载并缓存 tokenizer，保证一个进程内只加载一次。

    为什么需要缓存：
      AutoTokenizer.from_pretrained 每次调用都会读文件、构建词表，比较慢。
      分块函数可能被反复调用（比如一次索引多篇论文），用 lru_cache 缓存后，
      第二次起直接复用同一个对象，避免重复开销。

    性能隐患：
      第一次调用会读取本地模型目录里的 tokenizer 文件（几百 KB 到 1 MB），
      如果路径不存在或文件不全，会在这里抛异常。部署时（第 5 块）要确保
      models/bge-small-en-v1.5 目录随镜像一起打包。
    """
    return AutoTokenizer.from_pretrained(TOKENIZER_NAME)


def _extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """把 PDF 逐页抽成 [(页码, 该页文字)]，页码从 1 开始。

    参数：
      pdf_path : PDF 文件路径（字符串）。

    返回：
      列表，每项是 (页码, 该页纯文本)。文字为空的页也会保留，文字记为 ""。

    错误语义（契约 6，本函数不吞异常）：
      - 文件不存在：pdfplumber.open 抛 FileNotFoundError，直接向上抛。
      - 文件不是合法 PDF：pdfplumber.open 抛解析异常，直接向上抛。
      由上层（第 3 块的 API 层）决定怎么转成用户可见的错误提示。
    """
    pages = []
    # 用 with 打开，确保文件句柄用完自动关闭，避免 Windows 上文件被占用删不掉。
    with pdfplumber.open(pdf_path) as pdf:
        # enumerate(..., start=1)：页码从 1 开始，符合人的阅读习惯，
        # 也方便后面直接写进引用（"第 4 页"而不是"索引 3"）。
        for i, page in enumerate(pdf.pages, start=1):
            # extract_text() 对扫描件 / 无文字层的页可能返回 None，
            # 用 `or ""` 统一成空字符串，避免后面拼接时报 NoneType 错误。
            pages.append((i, page.extract_text() or ""))
    return pages


# 章节标题的正则，逐段解释：
#   ^\s*            行首，允许有前导空格（PDF 抽取常带缩进）
#   \d+(?:\.\d+)*   数字编号，支持 "3"、"3.1"、"3.1.2" 这种多级编号
#   \s+             编号和标题之间至少一个空白
#   [A-Z]           标题必须以大写字母开头（排除 "3 apples" 这类正文句子）
#   [^\n]{0,80}     标题剩余部分，最多 80 个字符，不跨行
#   $               行尾
#   re.MULTILINE    让 ^ 和 $ 匹配每一行的开头结尾，而不只是整段文本的首尾
_SECTION_RE = re.compile(r"^\s*\d+(?:\.\d+)*\s+[A-Z][^\n]{0,80}$", re.MULTILINE)


def _detect_section(text: str) -> str:
    """在一段 chunk 文本里找第一个「数字 + 大写标题」行，作为章节名。

    参数：
      text : 一个 chunk 的正文。

    返回：
      匹配到的标题行（去掉首尾空白）；没匹配到返回 ""。

    语义边界（重要，见 DESIGN.md 决策 5）：
      它返回的是「这个 chunk 内部出现过的第一个章节标题」，
      而不是「这个 chunk 整体属于哪一节」。因为 chunk 可能跨节，
      而 PDF 里章节的层级关系很难可靠还原，所以只做「能带就带」。
    """
    # search 只找第一个匹配，符合「块内第一个标题」的语义。
    m = _SECTION_RE.search(text)
    return m.group(0).strip() if m else ""


def parse_and_chunk(pdf_path: str, chunk_size: int = 400, overlap: int = 80) -> list[Chunk]:
    """把一篇 PDF 解析并切成带出处的文本块列表。

    参数：
      pdf_path   : PDF 文件路径。
      chunk_size : 每块的目标 token 数，默认 400。
      overlap    : 相邻块重叠的 token 数，默认 80。

    返回：
      Chunk 列表。空 PDF / 扫描件返回空列表 []。

    错误：
      - overlap >= chunk_size 时抛 ValueError（否则滑窗步长 <= 0 会死循环）。
      - 文件不存在 / 非法 PDF 的异常由 _extract_pages 向上抛，本函数不捕获。

    算法（四步）：
      1. 逐页抽文字，拼成全文，同时记录「每个字符属于第几页」。
      2. 用 tokenizer 把全文编码成 token 序列，并拿到每个 token 对应的字符区间。
      3. 按 chunk_size 滑窗（步长 = chunk_size - overlap）切 token 窗口。
      4. 每个 token 窗口映射回字符区间，切出原文，回填页码、章节、chunk_id。
    """
    # 防御：步长必须为正，否则 while 循环永远前进不了。这是唯一一处主动校验。
    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    pages = _extract_pages(pdf_path)

    # 【核心逻辑】
    # 把各页文字拼成一整段 full_text，并用 char_pages 记录「每个字符来自第几页」。
    # 为什么要字符级页码映射？因为切块是按 token 切的，块可能跨页；
    # 只有知道每个字符的页码，切完块才能反查这个块从第几页到第几页。
    parts: list[str] = []
    char_pages: list[int] = []
    for page_num, text in pages:
        if not text:
            continue  # 空页直接跳过，不往全文里塞空字符串
        if parts:
            # 页与页之间补一个换行，避免上一页末尾和下一页开头粘成一个词。
            # 【假设特定输入格式】这个换行符的页码记成「后一页」，影响极小。
            parts.append("\n")
            char_pages.append(page_num)
        parts.append(text)
        # 该页每个字符都打上页码标签；len(text) 个字符对应 len(text) 个页码。
        char_pages.extend([page_num] * len(text))
    full_text = "".join(parts)

    # 契约 5a：解析成功但没有任何文字（空 PDF / 扫描件）-> 返回空列表，不报错。
    # 是否提示用户「这是扫描件」由上层决定，本模块不负责。
    if not full_text.strip():
        return []

    tokenizer = _get_tokenizer()
    # 【性能隐患】
    # 把整篇论文一次性编码，内存峰值随论文长度上升；单篇论文（几万 token）可接受。
    # add_special_tokens=False：不要加 [CLS]/[SEP]，我们只关心原文 token 的位置。
    # return_offsets_mapping=True：返回每个 token 在原文中的 [字符起, 字符止)。
    # truncation=False：不要截断，否则长文会被悄悄砍掉，导致后半篇丢失。
    #
    # 注意：这里会打印 warning「sequence length 26950 > 512」。
    # 那是 BGE 模型「前向计算」的上限，而我们只是用 tokenizer 算字符位置，
    # 并没有把整篇喂给模型；每个 chunk 是 400 token，在 512 以内，不影响后续嵌入。
    enc = tokenizer(
        full_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )
    offsets = enc["offset_mapping"]  # 形如 [(0, 3), (3, 9), ...]，每个 token 的字符区间

    doc_name = os.path.basename(pdf_path)   # 只取文件名，如 "sample.pdf"
    step = chunk_size - overlap             # 滑窗步长；400/80 -> 320
    chunks: list[Chunk] = []
    start = 0
    while start < len(offsets):
        end = min(start + chunk_size, len(offsets))  # 最后一个窗口可能不满 400
        # 【核心逻辑】
        # token 窗口 [start, end) -> 字符区间 [char_start, char_end) -> 原文切片。
        # 用 offsets[start][0] 取窗口第一个 token 的字符起点，
        # 用 offsets[end-1][1] 取窗口最后一个 token 的字符终点（end-1 是因为右开区间）。
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        text = full_text[char_start:char_end]
        if text.strip():  # 跳过纯空白块，避免产生无意义的 chunk
            # 用字符级页码映射回填这个块的起止页：
            #   起页 = 块第一个字符所在页；止页 = 块最后一个字符所在页。
            page_start = char_pages[char_start]
            page_end = char_pages[char_end - 1]
            section = _detect_section(text)
            # chunk_id 规则见 Chunk 类注释；len(chunks) 是「已生成块数」，
            # 作为文档内递增序号，保证同一文档内不重复。
            chunk_id = f"{doc_name}#p{page_start}-{page_end}#c{len(chunks)}"
            chunks.append(Chunk(text, doc_name, page_start, page_end, section, chunk_id))
        if end == len(offsets):
            break  # 已经到最后一个 token，结束，避免多余的一次循环
        start += step
    return chunks