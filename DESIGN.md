# PaperLens 设计文档

本文件存档每个模块的**契约**与**设计决策**，以及实现过程中发生的**变更**。
规则：契约定死后改动视为需求变化，需确认；设计可演进，但演进必须记录原因。

---

## 第 1 块：PDF 解析 + 分块

### 契约（已确认）

**输入**
- 单个本地 PDF 路径。不收 URL、不收非 PDF、不做批量。

**输出**
- `list[Chunk]`，每个 Chunk 六个字段：

| 字段 | 类型 | 含义 | 可空 |
|---|---|---|---|
| `text` | str | 块正文 | 否 |
| `doc_name` | str | 文档文件名 | 否 |
| `page_start` | int | 起始页（从 1 开始） | 否 |
| `page_end` | int | 结束页（≥ page_start） | 否 |
| `section` | str | 块内出现的第一个章节标题 | 是（`""`） |
| `chunk_id` | str | 全局唯一 ID | 否 |

**分块参数**
- `chunk_size=400` tokens，`overlap=80` tokens，作为带默认值的函数参数，可覆盖。

**token 计数口径**
- 使用 `BAAI/bge-small-en-v1.5` 的 tokenizer。

**边界行为**
- 5a 空 PDF / 扫描件：返回 `[]`，不报错。
- 5b 单页文字短于块大小：整页成块。
- 5c 允许跨页，`page_start`/`page_end` 记录起止页。
- 5d 不做页眉页脚 / 参考文献过滤。
- 5e 章节识别失败：`section=""`，不中断。

**错误语义**
- 文件不存在 → `FileNotFoundError`。
- 非法 PDF → 解析库异常向上抛，不吞。
- 本块不捕获、不包装、不打日志。

**接口**
- `parse_and_chunk(pdf_path, chunk_size=400, overlap=80) -> list[Chunk]`

**验收标准**
- 8a 真实论文返回非空列表。
- 8b 字段合法（text 非空、doc_name 正确、page_start≥1、page_end≥page_start）。
- 8c `chunk_id` 全局唯一。
- 8d 覆盖率人工抽查（已执行通过）。
- 8e 文件不存在抛 `FileNotFoundError`。
- 8f 非法 PDF 抛异常。

### 设计决策（8 条）

| # | 决策 | 原因 | 被放弃的替代方案及代价 |
|---|---|---|---|
| 1 | 解析库用 **pdfplumber** | 双栏论文抽取顺序比 pypdf 稳，纯 Python，许可证干净 | pypdf：快但双栏易乱、表格不可用；PyMuPDF：质量最好但 AGPL，有法务顾虑 |
| 2 | `Chunk` 用 **dataclass** | 字段固定，有名字、可读、拼错字段名会报错 | plain dict：零依赖但字段名靠字符串，拼错运行到后面才发现 |
| 3 | tokenizer 用 **transformers AutoTokenizer**，进程内缓存 | 与后续嵌入选型同源，`lru_cache` 避免重复加载 | 字符数近似：精度差；tiktoken：口径与 BGE 不同，违背契约 |
| 4 | **按页解析 → 字符流 + 页码映射 → token 滑窗 → 回填页码** | 切窗后能准确回填起止页，天然支持跨页 | 按页切：超长页超限、跨页句子被切断；按章节切：章节识别不可靠 |
| 5 | `section` **能识别就带，识别不到填 `""`**，正则匹配「数字 + 标题」 | 章节是锦上添花，不能为它让主流程失败 | 用 LLM/版式分析识别：慢、有依赖；完全不填：放弃低成本信息 |
| 6 | `chunk_id` = `{doc_name}#p{起页}-{止页}#c{序号}` | 可读、可追溯、文档内唯一，无需 hash 库 | UUID/hash：绝对唯一但不可读，重复解析 ID 会变，不利于复现 |
| 7 | **单文件** `ingest.py`，四函数 | 本块内聚、体量小，最小实现 | 拆多文件：当前规模过度设计 |
| 8 | 依赖 **pdfplumber、transformers、torch(CPU)、pytest** | 契约要求的最小集 | 用 sentence-transformers 间接带 tokenizer：更重且本块用不到模型 |

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C1 | 决策 3：tokenizer 从 HF 仓库名改为**本地路径** `models/bge-small-en-v1.5` | 当前网络无法稳定访问 HF / hf-mirror，改为离线加载 |
| C2 | 依赖：新增 `models/` 本地目录 | 离线方案需要本地模型文件 |
| C3 | 决策 5 语义精确化：`section` = **块内出现的第一个章节标题**，不保证整块归属该节 | 实现用 `re.search` 找块内第一个匹配，原描述「所属章节」有歧义 |
| C4 | 已知限制新增：单词间空格可能丢失；整篇编码触发 tokenizer warning（无害） | 8d 人工抽查发现，记录为已接受边界 |
| C5 | `ingest.py` 顶部补 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1` | 与 `test_ingest.py` 保持一致，防止将来误改路径时联网卡死 |

---

## 第 2 块：嵌入 + 向量库索引

### 契约（已确认）

**前置条件**
- 嵌入模型本体（约 130MB）离线可用，放在 `models/bge-small-en-v1.5/`，与第 1 块 tokenizer 同目录。

**建索引 build_index(chunks)**
- 输入：`list[Chunk]`。空列表建空索引，不报错。
- 存储：Chroma 嵌入式，`PersistentClient(path="data/chroma")`。
- 共用一个 Collection `"papers"`，用 metadata 的 `doc_name` 区分文档。
- 幂等：同一文档重复建索引，先按 `doc_name` 删除旧数据，再写入。
- 分批嵌入（每批 32），向量 L2 归一化，距离度量 cosine。
- 每条记录：`id = chunk_id`，`document = chunk.text`，metadata 含 `doc_name / page_start / page_end / section / chunk_id`。

**检索 search(query, top_k=8)**
- 输入：`query: str`，`top_k: int = 8`（必须 ≥ 1）。
- 空 query 返回 `[]`；空索引返回 `[]`。
- 输出：`list[RetrievedChunk]`，按 `score = 1 - cosine_distance` 降序，长度 ≤ top_k。**必须返回分数**（第 7 块拒答阈值依赖它）。

**清空 clear_index()**
- 删除 Collection 后重建，保证名字不变。

**边界行为**
- 8a 空索引检索返回 `[]`。
- 8b 模型文件缺失抛异常，不吞。
- 8c 重复建索引幂等。
- 8d 空 query 返回 `[]`。

**错误语义**
- 模型缺失、Chroma 目录不可写：抛异常，不吞、不包装、不打日志。

**接口**
- `build_index(chunks: list[Chunk]) -> None`
- `search(query: str, top_k: int = 8) -> list[RetrievedChunk]`
- `clear_index() -> None`

**验收标准**
- 11a 建索引不报错，Collection 条数等于 chunk 数。
- 11b 相关问题 top_k=3 返回 3 条，score 递减。
- 11c 第 1 条结果的 `page_start` 落在第 1–2 页。
- 11d 无关问题最高分明显低于相关问题最高分（不写死阈值）。
- 11e 重复建索引两次，条数不变。
- 11f 空索引检索返回 `[]`。
- 11g 空 query 返回 `[]`。

### 设计决策（10 条）

| # | 决策 | 原因 | 被放弃的替代方案及代价 |
|---|---|---|---|
| 1 | 嵌入库用 **sentence-transformers** | BGE 是 ST 格式，一行加载 + 一行完成分批/归一化；官方推荐，坑少 | transformers 手写 mean pooling：可控但池化写错向量就废；代价是省一个包但增出错风险 |
| 2 | Chroma **PersistentClient + lru_cache 单例** | 持久化、避免重复 open 的锁冲突 | 内存 Client：重启丢数据；Chroma Server：多一个服务，免费单容器负担 |
| 3 | `get_or_create_collection("papers", {"hnsw:space": "cosine"})` | 幂等创建，距离度量与契约一致 | 默认 L2：不配但分数含义不同，违背契约 |
| 4 | 幂等 = **先按 doc_name 删，再写入** | 能处理「新 chunks 变少」时旧数据残留的问题 | upsert 按 id 覆盖：代码短但残留过期数据，隐蔽 bug |
| 5 | 模型 `@lru_cache` 单例 + `_embed` 封装 | 只加载一次；建索引和检索共用同一套编码逻辑 | 每次 new 模型：延迟爆炸；两处各写 encode：归一化参数易不一致 |
| 6 | 返回 `RetrievedChunk(chunk, score)`，`score = 1 - distance` | 相似度方向符合直觉，第 7 块设阈值不用反向写 | 返回原始 dict：上层耦合 Chroma 结构；返回 distance：阈值方向反直觉 |
| 7 | search 开头**显式判空** | 契约 8a/8d；避免空 query 白跑一次嵌入 | 交给 Chroma 自然返回：空 query 浪费 CPU，空库行为依版本 |
| 8 | `clear_index` = **删 Collection 后重建** | 最干净，空 where 的 delete 跨版本行为不一致 | `delete(where={})`：保留元数据但行为不稳 |
| 9 | **单文件** `index_store.py` | 与第 1 块同风格，内聚、最小实现 | 拆 embedder + store：当前规模过度设计 |
| 10 | 依赖 **chromadb、sentence-transformers**，模型复用 `models/bge-small-en-v1.5/` | 契约要求离线；复用目录避免两份模型 | 另下一份 ST 格式模型：目录干净但重复占空间 |

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C6 | 新增 `clear_index()` 对外接口 | 评估和调试需要清空索引，契约 10 已确认 |
| C7 | 幂等删除假设「一批 chunks 同属一篇论文」 | 实现用第一条的 `doc_name` 做删除依据；混合文档场景不成立，已写入 README 已知限制 |
| C8 | 已知限制新增：纯向量检索区分度有限，top-8 中仅 1 条直接含答案 | 第 2 块验收时人工诊断发现，作为 P1 重排的基线与动机记录 |

---

## 第 3 块：检索 + 生成 + 引用

### 契约（已确认）

**平台与调用方式**
- 平台：智谱开放平台 `bigmodel.cn`，模型 `glm-4.7-flash`，完全免费（输入输出均不计费）。
- 端点：`https://open.bigmodel.cn/api/paas/v4/`（OpenAI SDK 兼容格式）。
- API Key：从环境变量 `ZHIPU_API_KEY` 读取，不进代码。
- 调用方式：串行，绝不并发（免费档并发限制为 1）。
- 超时：单次调用 60 秒。
- 重试：遇限流/过载（1302/1305/429）等 10 秒重试一次；仍失败则抛异常。不写复杂退避算法。

**接口**
- `answer(query: str, top_k: int = 8) -> RagAnswer`
- 内部调用 `index_store.search()`，不感知检索细节。

**输出**
- `RagAnswer(answer, citations, usage)`：
  - `answer`：LLM 生成文本；拒答时为 `REFUSAL_TEXT`。
  - `citations`：`Citation` 列表，含 `chunk_id / doc_name / page_start / page_end / text_snippet`（截 200 字符）。
  - `usage`：token 用量 dict，P0 能读到就填，读不到为 `None`。

**检索为空**
- 不调 LLM，直接返回 `answer=REFUSAL_TEXT`、`citations=[]`。

**Prompt 结构**
- system 用英文写规则：只依据资料、不编造、用 `[n]` 标注引用、无资料说「未提及」、**用与问题相同的语言作答**。
- user 用英文拼资料（`[n] (page X)` + 原文），末尾附用户问题原文。
- `temperature=0.1`，`max_tokens=1024`。

**引用解析**
- 正则 `\[(\d+)\]` 提取编号 → 去重 → 按 1-based 映射回检索结果 → 越界忽略。
- 一个都没解析到则**降级返回全部检索结果**，请求不失败。

**错误语义**
- Key 缺失抛 `ValueError`；API 调用失败抛异常，不吞、不包装、不打日志。

**验收标准**
- 9a 正常问题返回非空 `answer` 和非空 `citations`。
- 9b 答案中至少提到一个关键数字（`10,000` / `0.01%` / `350GB` / `35MB`）。
- 9c 空 query 返回拒答文本，不调 LLM。
- 9d 清空索引后正常 query 返回拒答文本，不报错。
- 9e 未设 `ZHIPU_API_KEY` 时抛 `ValueError`。
- 9f 人工检查：答案含 `[n]`，`citations` 能对应到页码（已执行；发现引用可能错位，记录为已知限制）。

### 设计决策（10 条）

| # | 决策 | 原因 | 被放弃的替代方案及代价 |
|---|---|---|---|
| 1 | 用官方 **`zhipuai` SDK** | 官方维护，参数与 OpenAI 风格一致，坑少 | `openai` 兼容层：省一个包但错误码/流式细节不一定对齐；`requests` 手写：代码量翻倍 |
| 2 | **单文件** `qa.py`，两 dataclass + 四函数 | 与第 1、2 块同风格，纯字符串处理拆私有函数便于测试 | 拆 prompt/llm/citations 三文件：当前规模过度设计 |
| 3 | Client **`@lru_cache` 单例**，Key 缺失转 `ValueError` | 复用连接池；Key 检查集中一处，提示明确 | 每次新建 Client：慢；Key 检查散落各处 |
| 4 | 资料格式 **纯文本 `[n] (page X)` + 原文** | LLM 对纯文本编号的引用标注比 JSON 更稳、token 更少 | JSON 结构：更清晰但引用标注不稳、token 更多；不带页码：引用质量下降 |
| 5 | 引用解析：**正则提取 → 去重 → 越界忽略 → 全失败降级返回全部** | 简单可靠；降级保证解析失败不让请求失败 | 要求 LLM 输出 JSON：解析失败率更高；不解析直接返回全部：引用与答案对不上 |
| 6 | `text_snippet` 截 **200 字符** | 引用面板只需「一眼看出是哪段」；完整原文传输浪费 | 完整原文：引用列表臃肿；50 字符：常截在句子中间 |
| 7 | 重试收敛在 `answer()`，限流类错误 `sleep(10)` 重试一次 | 契约要求；集中一处不散落 | 不重试：评估时偶发 429 丢样本；指数退避：违反最小实现 |
| 8 | `usage` P0 能读到就填，读不到 `None` | 读一下成本近零，P2 做日志直接可用 | P0 完全不碰：P2 要改结构，无谓返工 |
| 9 | 新增依赖 **`zhipuai`**（+ 手动补 `sniffio`） | 官方 SDK；`sniffio` 是 zhipuai 漏声明的依赖 | 只用 `openai`：兼容层不确定性 |
| 10 | 语言规则：system 写死「同语言作答」，user 附问题原文 | 最直接，不需要语言检测依赖 | 代码里检测语言再拼指令：多一个依赖/正则，LLM 本身就能遵循指令 |

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C9 | 已知限制新增：LLM 引用可能错位（机制正常，标注不精确） | 9f 人工检查发现，记录为 RAG 已知问题，由第 7 块 Faithfulness 量化 |
| C10 | `_call()` 增加 `stream=False` 与返回类型注解 `-> Any` | zhipuai 类型 stub 未按 `stream` 重载，Pylance 误报 `choices` 未知属性；运行时行为经 5 passed 验证不变 |
| C11 | 决策 3 适配 Gradio 6.x：移除 `Chatbot(type="messages")` 参数 | Gradio 6.0 起 `type` 参数被移除，messages 格式成为唯一默认格式 |
| C12 | 已知限制新增：多用户共存于同一 Collection、上传文件不持久、无流式输出、消息旁显示耗时 | 第 4 块验收时确认的 P0 边界 |
| C13 | 已知限制新增：Gradio 6.x 的 `processing | x/y s` 进度指示器 | 验收截图确认，属官方 UI 特性，无需改代码 |
| # | 变更 | 原因 |
|---|---|---|
| C14 | 开启 Gradio `queue()`，将 `zhipuai` 超时从 60s 放宽至 120s | 解决中文提问偶发 `Request timed out` 的问题，确保长耗时推理不中断 |