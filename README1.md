---
# 详细文档见https://modelscope.cn/docs/%E5%88%9B%E7%A9%BA%E9%97%B4%E5%8D%A1%E7%89%87
domain: #领域：cv/nlp/audio/multi-modal/AutoML
# - cv
tags: #自定义标签
-
datasets: #关联数据集
  evaluation:
  #- iic/ICDAR13_HCTR_Dataset
  test:
  #- iic/MTWI
  train:
  #- iic/SIBR
models: #关联模型
#- iic/ofa_ocr-recognition_general_base_zh

## 启动文件(若SDK为Gradio/Streamlit，默认为app.py, 若为Static HTML, 默认为index.html)
# deployspec:
#   entry_file: app.py
license: Apache License 2.0
---
#### Clone with HTTP
```bash
 git clone https://www.modelscope.cn/studios/oceanlh/PaperLens-RAG.git

# PaperLens —— 学术论文 RAG 助手

上传学术论文 PDF，基于原文回答技术问题，并给出**页码级引用**；检索不到证据时**明确拒答**，而不是强行生成。

> 当前进度：**第 1–3 块已完成并通过验收**（解析分块 → 向量索引 → 检索生成引用）。后续模块见「路线图」。

---

## 目录结构

```
PaperLens/
├── data/
│   ├── sample.pdf        # 样例论文（LoRA），用于测试与演示
│   └── chroma/           # Chroma 持久化目录（第 2 块自动生成）
├── models/
│   └── bge-small-en-v1.5/ # 离线模型（tokenizer + 权重 + ST 配置）
├── ingest.py             # 第 1 块：PDF 解析 + 分块
├── test_ingest.py        # 第 1 块验收测试
├── index_store.py        # 第 2 块：嵌入 + 向量库索引
├── test_index_store.py   # 第 2 块验收测试
├── qa.py                 # 第 3 块：检索 + 生成 + 引用
├── test_qa.py            # 第 3 块验收测试
├── README.md             # 本文件
└── DESIGN.md             # 契约、设计决策、变更记录
```

---

## 环境要求

- Windows 10 / 11（本项目在 Win10 + Python 3.13 上开发）
- Python 3.13
- 依赖：
  - 第 1 块：`pdfplumber`、`transformers`、`torch`（CPU 版）、`pytest`
  - 第 2 块：`chromadb`、`sentence-transformers`
  - 第 3 块：`zhipuai`、`sniffio`（zhipuai 漏声明的依赖，需手动补装）

### 首次配置

1. 创建并激活虚拟环境（VS Code 通常会自动创建 `.venv`）：

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   & d:\python\PaperLens\.venv\Scripts\Activate.ps1
   ```

2. 配置 pip 国内镜像（只需一次）：

   ```powershell
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. 安装依赖：

   ```powershell
   pip install pdfplumber transformers torch pytest chromadb sentence-transformers zhipuai sniffio -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. 设置智谱 API Key（第 3 块需要；只在当前 PowerShell 窗口有效，重开终端需重设）：

   ```powershell
   $env:ZHIPU_API_KEY = "你的key"
   ```

   验证：`echo $env:ZHIPU_API_KEY` 能看到值即可。不要把 key 写进代码。

### 离线模型

本项目不联网加载模型。模型文件已提前下载到 `models/bge-small-en-v1.5/`，`ingest.py` 和 `index_store.py` 直接读本地路径。

该目录需要完整模型（tokenizer 文件 + 模型权重 + sentence-transformers 配置，约 130MB），不只是第 1 块用的 tokenizer 小文件。

如果换一台机器、这个目录不存在，需要重新获取（任选其一）：

- ModelScope（推荐）：

  ```powershell
  pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple
  ```

  ```python
  from modelscope import snapshot_download
  snapshot_download('BAAI/bge-small-en-v1.5', local_dir='./models/bge-small-en-v1.5')
  ```

- 手动下载：从 hf-mirror.com 下载完整模型目录，放进 `models/bge-small-en-v1.5/`。

## 运行

### 第 1 块：解析与分块

```powershell
pytest test_ingest.py -v
```

预期：5 passed（`data/sample.pdf` 存在时）。

手动查看分块结果：

```powershell
python -c "from ingest import parse_and_chunk; cs = parse_and_chunk('data/sample.pdf'); print('chunks:', len(cs)); [print(f'--- {c.chunk_id} | p{c.page_start}-{c.page_end} | section={c.section!r} | {len(c.text)} chars ---') for c in cs[:3]]"
```

### 第 2 块：建索引与检索

```powershell
pytest test_index_store.py -v
```

预期：7 passed。注意：test_11f 会清空索引，所以测试跑完后索引为空。

要手动体验检索，先重建索引：

```powershell
python -c "from ingest import parse_and_chunk; from index_store import build_index; build_index(parse_and_chunk('data/sample.pdf')); print('index built')"
```

然后检索：

```powershell
python -c "from index_store import search; rs = search('How much does LoRA reduce trainable parameters?', top_k=3); [print(f'{r.score:.4f} | {r.chunk.chunk_id} | p{r.chunk.page_start} | {r.chunk.text[:120]!r}') for r in rs]"
```

预期：3 条结果，score 从高到低递减。

### 第 3 块：检索 + 生成 + 引用

```powershell
pytest test_qa.py -v
```

预期：5 passed（需已设置 `$env:ZHIPU_API_KEY`；未设置时 9a/9b 会 SKIPPED，其余 3 个仍通过）。

手动查看一次完整问答：

```powershell
python -c "from qa import answer; r = answer('How much does LoRA reduce trainable parameters?'); print('=== ANSWER ==='); print(r.answer); print(); print('=== CITATIONS ==='); [print(f'[{i+1}] {c.doc_name} p{c.page_start}-{c.page_end} | {c.chunk_id}\n     {c.text_snippet[:100]!r}') for i, c in enumerate(r.citations)]; print(); print('=== USAGE ==='); print(r.usage)"
```

## 已知限制

这些是有意接受的边界，不是 bug，详见 DESIGN.md。

### 第 1 块

- PDF 双栏排版可能导致文字顺序错乱；公式、表格抽取不保证正确。
- 单词间空格可能丢失（如 Edward Hu → EdwardHu）。因为检索靠语义向量而非关键词，暂不修复；若评估显示影响召回再处理。
- 章节识别不保证成功，section 可能为空；且它表示「块内出现过的第一个标题」，不保证整块归属该节。
- 页眉页脚、参考文献未过滤，会进入索引。
- 整篇编码时 tokenizer 会打印 sequence length > 512 的 warning，属正常现象（只是算字符位置，不喂给模型）。

### 第 2 块

- `build_index` 假设一批 chunks 属于同一篇论文（用第一条的 doc_name 做幂等删除）。传入混合文档时幂等逻辑不成立。
- `score = 1 - cosine_distance` 是相对排序依据，不是概率。拒答阈值必须在第 7 块实测后确定，不能拍脑袋定 0.5。
- `search` 的 top_k 没有上限校验；若传入大于库内条数的值，Chroma 会返回实际存在的条数（≤ top_k），不报错。
- 嵌入模型首次加载约几秒到十几秒；进程内缓存后不再重复加载。
- 纯向量检索对「语义相关但未直接回答」的段落区分度有限。实测「LoRA 减少多少参数」top-8：第 1 名（p2，score 0.7765）讲的是「减少存储开销」，真正含数字的 p5「10,000×」排第 3（0.7643），摘要的「reduce by 10,000 times」和 p3 的「0.01%」未进 top-8；8 条分数集中在 0.750–0.777 区间。P0 仍可用（top-8 含 1 条直接答案），但这是 P1 引入 Cross-Encoder 重排的直接动机，也是重排的对比基线。

### 第 3 块

- LLM 引用可能错位：实测「LoRA 减少多少参数」，答案正确写了 10,000 times，但标注的 [4] 指向 p1 标题段（不含该数字），真正含数字的 p5 c15 排第 3 未被引用。机制本身正常（解析、映射、降级都对），是 LLM 标注不精确。这是 RAG 已知问题，由第 7 块 Faithfulness 评估量化，P1 重排有望缓解。
- `answer()` 的重试只覆盖限流/过载类错误一次；其他错误（如网络中断）直接抛，不重试。
- 免费档 completion_tokens 可能高于可见答案长度（实测答案一句话，completion 677 tokens），属平台计费口径，不影响功能。

## 路线图

| 块 | 内容 | 状态 |
| --- | --- | --- |
| 1 | PDF 解析 + 分块 | ✅ 完成 |
| 2 | 嵌入 + 向量库索引（Chroma） | ✅ 完成 |
| 3 | 检索 + 生成 + 引用（GLM-4.7-Flash） | ✅ 完成 |
| 4 | Gradio 界面 | ⬜ 待开始 |
| 5 | Docker 化 + HF Spaces 部署 | ⬜ |
| 6 | 两阶段检索（重排，P1） | ⬜ |
| 7 | 评估脚本 + 拒答机制（P1） | ⬜ |
| 8 | 日志 / CI / 架构图 / 消融（P2） | ⏸ 以后再说 |


### 第 4 块

- Gradio 6.x 的 Chatbot 默认在每条消息旁显示处理耗时，属 UI 行为，不影响功能。
- 上传的 PDF 存在 Gradio 的临时目录，进程重启即失效。每次会话内完成「上传→索引→问答」，不依赖跨会话持久化。
- 索引是全局的：多个用户上传不同论文会**共存**于同一 Collection，检索时跨论文混合。P0 接受，不做会话隔离。
- P0 不做流式输出，用户点发送后需等待几秒到十几秒，期间界面显示处理中。
- **偶发超时**：GLM 免费模型在高峰期生成中文长句时，偶尔会超过 60 秒。P0 已通过把 `zhipuai` 的读取超时放宽到 120 秒，并启用 Gradio 队列（`queue()`）来规避，但极端情况下仍可能受平台算力限制而失败。

