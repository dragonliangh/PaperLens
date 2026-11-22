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
- 嵌入模型本体离线可用，放在 `models/bge-small-en-v1.5/`，与第 1 块 tokenizer 同目录。

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
| C8 | 已知限制新增：纯向量检索区分度有限，top-8 里直接含答案的条目偏少 | 第 2 块验收时人工诊断发现，作为 P1 重排的基线与动机记录 |

---

## 第 3 块：检索 + 生成 + 引用

### 契约（已确认）

**平台与调用方式**
- 平台：智谱开放平台 `bigmodel.cn`，模型 `glm-4.7-flash`，完全免费（输入输出均不计费）。
- 端点：`https://open.bigmodel.cn/api/paas/v4/`（OpenAI SDK 兼容格式）。
- API Key：从环境变量 `ZHIPU_API_KEY` 读取，不进代码。
- 调用方式：串行，绝不并发（免费档并发限制为 1）。
- 超时：单次调用 60 秒（第 5 块调整为 120 秒，见 C14）。
- 重试：遇限流/过载（1302/1305/429）等 10 秒重试一次；仍失败则抛异常。不写复杂退避算法。**注：退避重试不保证兜住——免费模型存在账户级限流，实测连续三次重试仍全部 429（见 C44）。**

**接口**
- `answer(query: str, top_k: int = 20, final_k: int = 4) -> RagAnswer`（第 6 块由 `top_k=8` 改为 20，并新增 `final_k`；`qa.py:754` 即为此签名）
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
| C10 | `_call()` 增加 `stream=False` 与返回类型注解 `-> Any` | zhipuai 类型 stub 未按 `stream` 重载，Pylance 误报 `choices` 未知属性；运行时行为经 5 passed 验证不变。**注：已被 C38 取代——阶段 3 起统一走流式，`_call()` 固定 `stream=True`，全仓无 `stream=False` 调用路径** |
| C44 | 已知限制补充：免费模型存在**账户级限流**，退避重试后仍可能连续失败 | 2026-11-21 本机复测（venv 已与 `requirements.txt` 对齐）：同一问题连续三次（含间隔等待）均返回 429——`1305` 模型当前访问量过大 / `1302` 账户已达速率限制；而同一把 Key 的非流式调用可以成功，反证「配额可用，限流与请求形态、时段负载相关」。故「等 `RETRY_DELAY_SECONDS` 重试一次」不保证兜住，用户侧表现为「生成服务繁忙，请稍后重试」。属平台侧配额，非代码缺陷、非 Key 失效；与 README 第 3 块「LLM 生成」同一口径。 |

---

## 第 4 块：Gradio 界面

### 契约（已确认）

**技术选型**
- 界面库：Gradio（最终锁定 5.49.1，原因见 C15）。
- P0 范围：不异步、多用户隔离、自定义 CSS。流式输出与降级容错已于阶段 3–4 落地（见 C38 / C40 与 README「阶段 3–4」）。

**界面组成**
- 上传区：`gr.File`（只收 `.pdf`）+「上传并建索引」按钮。
- 状态区：`gr.Markdown`，显示索引结果或错误。
- 聊天区：`gr.Chatbot`（Gradio 5.x 需显式 `type="messages"`）。
- 输入区：`gr.Textbox` +「发送」按钮。
- 引用区：`gr.Markdown`。

**上传行为**
- 上传成功后清空聊天历史和引用区，状态区提示「已索引 N 个 chunk，来自 {doc_name}。对话已重置。」（`app.py:103`，同时返回 `status, [], [], ""`）。

**提问行为**
- 异常时错误显示在聊天区，页面不崩。

**接口**
- `build_app() -> gr.Blocks`

**验收标准**
- 12a–12g（启动、界面组成、建索引、问答、空输入、异常提示、引用格式）。

### 设计决策（摘要）

| # | 决策 | 原因 |
|---|---|---|
| 1 | 单文件 `app.py`，三个内部回调 | 与前 3 块同风格，内聚 |
| 2 | 上下堆叠布局 | 窄屏不挤压 |
| 3 | `gr.State` 存 `list[dict]`，`type="messages"` | Gradio 新格式 |
| 4 | 上传后清空历史 | 避免「引用还是旧论文、状态说新论文」的不一致 |
| 5 | 异常放聊天区 / 状态区 | 用户视线焦点 |

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C11 | 决策 3：Chatbot 必须显式 `type="messages"`（Gradio 5.49.1） | 5.x 默认为 tuples 格式，传 dict 列表不声明 `type` 会渲染失败（`app.py:236-237`）；曾按 6.x 尝试移除该参数，与锁定版本冲突，已保持 5.x 写法 |
| C12 | 已知限制新增：多用户共存于同一 Collection、上传文件不持久、消息旁显示耗时（流式已于阶段 3 落地）；另记 Gradio 5.49.1 的 `processing \| x/y s` 进度指示器属官方 UI 特性，无需改代码 | 第 4 块验收时确认的 P0 边界与验收截图观察 |

> 编号说明：原 C13（进度指示器条目）已并入 C12，编号不复用。

---

## 第 5 块：Docker 化 + 魔搭部署

### 契约（已确认）

**平台约束**
- 魔搭创空间 Docker SDK，监听 7860，`0.0.0.0`。
- `ms_deploy.json` 声明 `{"sdk": "docker", "app_port": 7860}`。

**模型与索引**
- 模型**不**打进镜像（`.dockerignore` 排除 `models/`，`Dockerfile` 的 `COPY . .` 因此带不进权重），启动时由 `app.py` 的 `snapshot_download` 从 ModelScope 下载；魔搭云端启动时联网下载成功，故线上表现为"可用"，但不等于"不下载"。离线部署须预先将 `models/` 挂载或拷贝进容器，否则会卡在下载。
- 启动时 `clear_index()` 清旧数据，再自动索引 `data/sample.pdf`。

**依赖**
- Dockerfile 单独从 PyTorch CPU 源安装 `torch==2.15.0`，剥离 Linux 默认附带的 CUDA 依赖，使镜像体积大幅缩小（仓库未留存构建日志，具体 GB 数不引用）。
- `pdfplumber==0.11.8`（要求 `Pillow>=9.1`）+ `Pillow==11.3.0` + `gradio==5.49.1`（要求 `Pillow<12.0`），三者相容；`pdfplumber 0.11.10` 要求 `Pillow>=12.2.0`，与 gradio 的 `<12.0` 无交集，故不采用。

**验收标准**
- 13a–13f（Running、界面、自动索引、冷启动、上传新 PDF、Secrets 配置）。

### 设计决策（摘要）

| # | 决策 | 原因 |
|---|---|---|
| 1 | `ms_deploy.json` 显式声明 SDK 和端口 | 魔搭必需 |
| 2 | `python:3.13-slim` + 补 `build-essential` / `libgomp1` | 体积小，且够 PyTorch 用 |
| 3 | 端口三处对齐（Dockerfile EXPOSE、ms_deploy app_port、app.py launch） | 避免魔搭反复重定向 |
| 4 | `.dockerignore` 排除 `.venv` / `__pycache__` / `data/chroma` | 减小镜像 |
| 5 | `torch` 单独从 CPU 源安装 | 剥离 CUDA 依赖，显著缩小镜像体积（未留构建日志，不引用具体 GB） |

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C14 | 开启 Gradio `queue()`，将 `zhipuai` 超时从 60s 放宽至 120s | 解决中文提问偶发 `Request timed out` 的问题，确保长耗时推理不中断 |
| C15 | 依赖版本：锁 `pdfplumber==0.11.8`（要求 `Pillow>=9.1`）、`Pillow==11.3.0`、降级 `gradio==5.49.1`（要求 `Pillow<12.0`） | 解决魔搭云端构建时 Pillow 版本冲突，以及 Gradio 与 Pillow 的版本兼容性问题。`pdfplumber 0.11.10` 要求 `Pillow>=12.2.0`，与 gradio 无交集，故锁 0.11.8 |
| C16 | Dockerfile 单独从 PyTorch CPU 源安装 `torch==2.15.0` | 剥离 Linux 下默认附带的 CUDA 依赖，镜像体积大幅缩小（未留构建日志，不引用具体 GB） |
| C17 | `qa.py` 加入性能计时日志 | 定位耗时瓶颈：后端耗时集中在重排与 LLM 两段（全程 CPU 推理、未用 GPU，耗时随机器与负载变化），前端渲染会额外增加等待感 |

---

## 第 6 块：Cross-Encoder 精排

### 契约（已确认）

**模型**
- `BAAI/bge-reranker-base`（中英双语，278M 参数）。
- 放弃 `ms-marco-MiniLM-L-6-v2`：纯英文，中文跨语言场景不适用。

**参数**
- 粗排 `top_k=20`，精排 `final_k=4`。
- 可覆盖。

**接口**
- `rerank(query, candidates, final_k=4) -> list[RetrievedChunk]`
- 返回的 `score` 字段替换为 Cross-Encoder 的 0~1 相关性分数，**不再是余弦相似度**。

**`qa.answer()` 流程**
- 改为 `search(top_k=20)` → `rerank(final_k=4)` → 拼 prompt。对外签名不变。

**验收标准**
- 14a–14e（返回 ≤ final_k、p5 进 top-4、分数 0~1、端到端可用、重排在 CPU 上为十几秒量级、随机器变化）。

### 设计决策（摘要）

| # | 决策 | 原因 | 放弃的替代 / 代价 |
|---|---|---|---|
| 1 | `sentence-transformers` 的 `CrossEncoder` | 与 SentenceTransformer 同库 | 手写 transformers 加载：代码量大 |
| 2 | `@lru_cache` 单例 | 模型体积大，重复加载延迟爆炸 | 每次 new：重复加载明显拖慢启动 |
| 3 | sigmoid 归一化到 0~1 | 第 7 块阈值需要 0~1 量纲 | 保留 logits：跨模型不可比 |
| 4 | 单文件 `rerank.py` | 与前三块同风格 | 拆包：过度设计 |
| 5 | `final_k` 截断在 `rerank` 内部 | 调用方不需要写切片 | 调用方截断：易忘 |
| 6 | `batch_size=4` | CPU 内存扛不住 32（未限制时单次重排会慢到不可用，未留压测日志，不引用具体耗时） | 默认 32：内存 Swap |

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C18 | `_call()` 增加 `stream=False` 与 `Any` 类型注解 | 消除 Pylance 报错 |
| C19 | 新增 `rerank.py` 实现 Cross-Encoder 精排，模型选型 `bge-reranker-base` | 纯向量检索区分度有限、命中条目少且排名靠后，需重排 |
| C20 | `rerank.py` 强制设置 `batch_size=4`，并加入 `torch.sigmoid` 归一化 | CPU 内存 Swap 问题 + 第 7 块阈值量纲 |
| C21 | 本地重生成干净的 `.safetensors` 覆盖云端损坏文件 | 云端曾出现 `.safetensors` header 损坏 + `.bin` 触发 PyTorch 2.6 `weights_only` 双重问题。safetensors 是纯张量格式，加载时不走 `torch.load`，从根本上绕开安全限制 |
| C22 | 移除所有针对 `torch.load` 的 monkey-patch 和 `use_safetensors=False` 参数 | 模型修复后不再需要拦截，恢复干净代码便于维护。**注：仅部分落地——`use_safetensors=False` 已无残留，但 `ingest.py:12-19` 的 `torch.load` monkey-patch 仍在（`model.safetensors` 已能正常加载，补丁未删）** |

---

## 第 7 块：评估脚本 + 拒答机制

### 契约（已确认）

**评估集 `eval_set.json`**
- 规模：140 条（英文 70 + 中文 70，id 严格 `001-070` 连续，中英一一对应）。
- 可答题：每种语言 50 道，`gold_page` 人工标注，`gold_keywords` 为答案中必须包含的关键词。
- 不可答题：每种语言 20 道，按 **2:2:1** 结构——8 道 unrelated（完全无关）、8 道 semi_related（半相关）、4 道 counterfactual（反事实）。
- 中文题：与英文题语义严格对应，`gold_page` / `gold_keywords` 与英文完全一致（**keywords 用英文原词**，因为检索池是英文原文）。

**指标口径**
- 粗排 Recall@5：`search()` 的 top-20 中，前 5 是否命中 `gold_page`。
- 粗排 MRR@20：`search()` 的 top-20 中，第一个命中的排名倒数。
- 精排 Recall@4：`rerank()` 的 top-4 中，前 4 是否命中。
- 精排 MRR@4：`rerank()` 的 top-4 中，第一个命中的排名倒数。
- 命中判定：`gold_page ∈ [page_start, page_end]`（区间判定，容忍跨页）。
- 精排指标分母：`n_fine_scored`（有精排结果的题数），不是可答题总数。
- 拒答准确率：不可答题中，`top_score < threshold` 的比例。
- 误拒率：可答题中，`top_score < threshold` 的比例。
- **中英文分开统计，不做综合平均。**

**阈值**
- 独立 `thresholds.json`，格式 `{"reject": {"default": 0.3, "en": 0.3, "zh": 0.18}}`。
- 缺失抛异常，避免用默认值悄悄跑错。
- `qa.py` 用中文字符正则检测 query 语言，按语言选阈值，缺 lang 时回退 `default`。

**拒答逻辑**
- `qa.answer()` 在重排后、拼 prompt 前判断：`results[0].score < threshold` → 返回 `REFUSAL_TEXT`，不调 LLM。

**诊断项**
- `coarse_recall_at_20`：粗排天花板。
- `fine_recall_at_4_gold_given`：给定粗排 top-20 命中，精排 top-4 保住的比例（EN 0.7273 / ZH 0.6757）。
- `fine4_lost_from_coarse4_rate`：粗排 top-4 已命中、却被重排挤出 top-4 的比例（纯重排丢失，EN 0.1481 / ZH 0.2727）。
- `window_truncation_rate`：粗排 5-20 命中、但没进精排 top-4 的比例（窗口截断，扩 K 才有救，EN 0.16 / ZH 0.18）。
- `fine_recall_at_20_gold_given`：粗排 20 条**全量重排**后 gold 是否仍在列表里——候选集不变，故恒为 1.0，仅作参考项。
- `false_reject_rate`：可答题被误拒的比例。
- `n_fine_scored`：精排有效题数。
- `coarse_rank_histogram` / `fine_rank_histogram`：粗排 / 精排命中排名直方图。

**接口**
- `evaluate.py`：`main()` 跑完整评估，打印表格 + 写报告；`--report` 改输出路径，`--details` 额外落盘逐题明细，`--zh-threshold` / `--en-threshold` 临时覆盖阈值。

**验收标准**
- 15a `eval_set.json` 140 条，id 连续。
- 15b `evaluate.py` 跑完不报错，输出表格 + `eval_report.json`。
- 15c en / zh 各自指标都有数字。
- 15d 阈值从 `thresholds.json` 读，缺失抛异常。
- 15e 拒答准确率按 category 细分。

### 设计决策（9 条）

| # | 决策 | 原因 | 放弃的替代 / 代价 |
|---|---|---|---|
| 1 | 扁平列表 + id 前缀区分语言 | 遍历方便，中英对照一目了然 | 按语言拆文件：多一次 IO + 15 行合并逻辑 |
| 2 | 区间判定命中 | 容忍跨页，避免 Recall 虚低 | 精确匹配 page_start：跨页 chunk 会判未命中，Recall 虚低（边界分类实测 100 题中 5 题为跨页误匹配） |
| 3 | MRR 未命中记 0 | 标准定义，可复现 | 跳过未命中：只统计命中题，MRR 虚高 |
| 4 | 评估脚本只调 `search` + `rerank`，不调 `qa.answer` | 快、不消耗 API、逻辑透明 | 端到端：140 题要跑十几分钟以上，且必然触发 429（免费档并发 1） |
| 5 | 阈值人工观察分布后定 | 样本量小（当时可答题 60 道），ROC 无统计意义 | `sklearn` 算 ROC：多 0.5 人天，小样本上过拟合 |
| 6 | `REFUSAL_THRESHOLD` 初始占位 0.3，评估后调 | 先给个中间值避免报错 | 初始 0 或 1：无参考意义 |
| 7 | `eval_report.json` 含时间戳、阈值、分布、诊断项 | 回归对比、阈值调整不用重跑 | 只输出四个百分比：看不出「为什么变了」 |
| 8 | 单文件 `evaluate.py`，五函数 | 与项目同风格，内聚 | 拆 `evaluation/` 包：当前规模过度设计 |
| 9 | 不新增依赖 | 指标计算是纯数学 | 引入 `sklearn`：多一个重依赖，与决策 5 冲突 |

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C23 | 评估集规模从 120 条扩到 140 条（可答 48→50，不可答 12→20） | 原 48+12 的除不尽问题；新方案 50+20 除尽，且不可答题按 2:2:1 分配为 8+8+4 |
| C24 | `en_012` 换题 | 原题 keywords 全是单字母，任何页都能命中，MRR 无意义；换成 `How does LoRA's number of trainable parameters scale with the rank r?`（gold_page=4，keywords `["linear", "r"]`） |
| C25 | `en_021` gold_page 9→10、`en_041` gold_page 5→4 | 结论句实际在 Section 7.1（page 10）；"memory and storage" 的概括在 Section 4.2（page 4） |
| C26 | 中文题 `gold_keywords` 统一用英文原词 | 中文题检索的是英文原文，keywords 是「在检索结果里找什么」的锚点。用中文译词在英文原文里匹配失败，中文指标失真 |
| C27 | `thresholds.json` 从单值改为按语言分设（en=0.3, zh=0.18） | 中文误拒率在单阈值阶段明显高于英文，原因是 BGE 单语模型跨语言匹配分数天然偏低。分语言设阈值后误拒率 0.34、拒答准确率 0.70（`eval_report.json`） |
| C28 | 阈值从 0.15 回调到 0.18 | 2026-11-22 用同一次遍历的逐题明细重做了正式对比：召回类指标两档完全相同，差异只在拒答——0.18 比 0.15 多拦 1 道半相关不可答题（semi_related 3/8→4/8，拒答 0.65→0.70），代价是多误拒 1 道可答题（16/50→17/50）；unrelated 与 counterfactual 两档不动。即"一换一"，选 0.18 依据是"漏答代价 > 拒答代价"的价值排序（旧文"损失 5%/12.5%、边际收益为负"的表述基于两份非同一批数据的报告，已作废） |
| C29 | `qa.py` 新增 `_detect_lang` 和 `_get_threshold` | 支持按语言选阈值；兼容旧格式（数字 → `{"default": 数字}`） |
| C30 | `evaluate.py` 的 `fine_recall_at_20_gold_given` 口径修正 | 原实现是"粗排 20 条全量重排后 gold 是否仍在列表里"，候选集没变 → 恒为 1.0，不能证明重排没丢答案（旧报告里的 0.7273/0.6757 其实是"给定粗排 20 命中、精排 top-4 保住率"）。2026-11-22 补三个 top-4 口径诊断项：`fine_recall_at_4_gold_given`（EN 0.7273 / ZH 0.6757，与旧值逐位一致）、`fine4_lost_from_coarse4_rate`（EN 0.1481 / ZH 0.2727）、`window_truncation_rate`（EN 0.16 / ZH 0.18）；恒 1.0 的旧项降级为参考项并列展示 |

---

## 第 7 块补充：对照组评估

### 背景

主评估集（140 题）测「方法在混合题型上的整体表现」，但包含 12 道边界案例（符号型、否定句、表格行标签、章节主旨、跨页误匹配、低信号发布句、关系型语义模糊），这些题方法天然不擅长。为了验证「方法在擅长领域确实强」，新增对照组。

### 契约（已确认）

**评估集 `ctrl_eval_set.json`**
- 规模：40 条（20 英 + 20 中，语义严格对应）。
- 结构：与 `eval_set.json` 一致，含 `id / lang / type / question / answerable / category / gold_page / gold_keywords`。
- 类型：T1 具体数值型、T2 定义型、T3 直接引用型、T4 步骤/配置型、T5 单页内对比型，每类 4 题。
- 全部可答（无不可答题）。
- 防自证：题目由项目所有者出（不看答案位置），gold 由开发伙伴标注。

**指标口径**
- 与第 7 块主指标一致：粗排 Recall@5 / MRR@20、精排 Recall@4 / MRR@4。
- 中英文分开统计。
- 按 `type`（T1–T5）分组统计，看「哪种题型最强」。
- 不测拒答指标（对照组全是可答题）。

**接口**
- `evaluate_ctrl.py`：`main()` 跑对照组评估，打印表格 + 写 `ctrl_eval_report.json`。

**验收标准**
- 16a `ctrl_eval_set.json` 有 40 条，结构与 `eval_set.json` 一致。
- 16b `evaluate_ctrl.py` 跑完不报错，输出表格 + `ctrl_eval_report.json`。
- 16c en / zh 各自指标都有数字。
- 16d 按 type 分组统计有 5 行（T1–T5）。

### 实测结果（EN）

| 指标 | EN | ZH |
|---|---|---|
| 可答题数 | 20 | 20 |
| 粗排 Recall@5 | 0.45 | 0.25 |
| 粗排 MRR@20 | 0.35 | 0.22 |
| 精排 Recall@4 | 0.65 | 0.45 |
| 精排 MRR@4 | 0.45 | 0.43 |

**按 type 分组（EN）**：

| type | Recall@5 | MRR@20 | Recall@4 | MRR@4 |
|---|---|---|---|---|
| T1 数值型 | 0.75 | 0.64 | **1.00** | 1.00 |
| T2 定义型 | 0.25 | 0.14 | 0.75 | 0.38 |
| T3 引用型 | 0.25 | 0.33 | 0.75 | 0.38 |
| T4 步骤/配置型 | 0.75 | 0.44 | 0.50 | 0.38 |
| T5 对比型 | **0.25** | 0.19 | **0.25** | 0.13 |

### 设计决策（4 条）

| # | 决策 | 原因 | 放弃的替代 / 代价 |
|---|---|---|---|
| 1 | 对照组 20 道题 × 2 语言 = 40 条 | 20 题粒度够看出「方法擅长 vs 不擅长」，40 条平衡工作量 | 30 题：更稳但工作量大；10 题：粒度太粗 |
| 2 | 防自证：分离出题与标答案 | 避免「出题人知道答案位置」 | 一人包办：有自证风险 |
| 3 | 按 type 分组统计 | 看「哪种题型最强」，比整体平均更有信息量 | 只看整体：掩盖内部差异 |
| 4 | 接受英文粗排 Recall@5 0.45 低于主评估集英文的 0.60 | 诚实呈现，暴露方法边界 | 改题面刷分：违背工程诚实 |

### 结论

- **方法在「找一个数字」上确实满分**（T1 精排 Recall@4 = 1.00），这是核心优势。
- **方法在「比较概念」上接近 0**（T5 双指标 0.25），这是明确弱点。
- **对照组整体 0.45 不低不高**，因为「事实型」内部有强弱分化。这个结论比「对照组 0.85 验证假设」更有价值——它说明我们对方法能力边界的认知本身需要数据校准。（0.85 为当初的假设值，非实测）

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C31 | `evaluate_ctrl.py` docstring 从「20 道题（en 10 + zh 10）」改为「20 道题（中英各一版，40 条记录）」 | 对照组规模最终定为 20 道题 × 2 语言 = 40 条记录，初版 docstring 误写为 20 条记录 |
| C32 | `ctrl_eval_set.json` 每条补 `answerable: true` 和 `category: "answerable"`；`evaluate_ctrl.py` 的 `load_ctrl_eval_set` 退回纯 `json.load` | 保持两个评估集 JSON 结构一致，避免在代码里补字段 |
| C33 | 对照组最终结论采用 B+C 组合：保留数据、按 type 分组呈现、诚实写「假设部分证伪」 | 不换题（不美化指标），通过分组暴露「T1 满分 T5 最弱」的方法能力分化 |
| C34 | 阶段 2 契约 18a 用例从 `rq=rv=8` 改为 `AdamW` | 诊断脚本遍历 sample.pdf 全部 84 个 chunk，含 `rq`/`rv` token 的 chunk 数为 0。根因是 PDF 对下标 `q`/`v` 的字符提取不可靠，`rq=rv=8` 未被完整抽出，BM25 只能靠高频 token `8` 命中、区分度被稀释。这是第 1 块「公式、表格抽取不保证正确」已知限制的具体表现，非 BM25 实现缺陷。改用正文词 `AdamW`（第 19 页 GPT-3 实验配置，PDF 可可靠抽出），契约 18a 意图（BM25 对精确 token 能命中）不变。 |

---

## 第 8 块：混合检索（阶段 2，已回滚）

### 背景

阶段 2 目标：在向量检索外加 BM25 关键词检索一路，用 RRF 融合，提升英文 Recall。
按分阶段流程，先落 2a（`hybrid_search.py` + `test_hybrid.py` + `qa.py` 接入），再落 2b（`evaluate_hybrid.py` 量化），最后按数据决定保留或回滚。

**结论：实测无净提升，已回滚接入，代码保留作多文档场景储备。**

### 契约（已确认，2a）

**接口（`hybrid_search.py`）**
- `bm25_search(query, top_k=20) -> list[Chunk]`：英文 BM25 检索，中文返回 `[]`，空输入/空索引返回 `[]`，不带分数。
- `search_hybrid(query, top_k=20) -> list[RetrievedChunk]`：英文 RRF 融合；中文透传 `index_store.search()`。
- `rebuild_bm25() -> None`：从 Chroma 读全部 chunk 重建内存 BM25 索引。

**RRF**
- `score = Σ 1/(60 + rank_i)`，按 `chunk_id` 去重，降序截 top_k。

**生命周期**
- 数据源：从 Chroma 读（1a）；建表时机：首次 `bm25_search` 懒加载（2a）；同步：显式 `rebuild_bm25()`（3b + 防法 1）。
- `search_hybrid` 对 `en` 返回 RRF 分数（约 0~0.033，**非余弦**），对 `zh` 返回余弦分数；**下游 rerank 必须覆盖，不得直接用于拒答阈值**。

**分词**
- 英文正则 `[a-z0-9]+`；中文不做 BM25，不引入 jieba。

**依赖**
- 新增 `rank_bm25`。

### 实测结果（`hybrid_eval_report.json` / `diag_hybrid_report.json`）

| 指标（EN） | A 纯向量 | B 混合 | Δ |
|---|---|---|---|
| 粗排 Recall@5 | 0.60 | 0.58 | -0.02 |
| 粗排 MRR@20 | 0.386 | 0.436 | +0.05 |
| 精排 Recall@4 | 0.64 | 0.64 | 0 |
| 精排 MRR@4 | 0.428 | 0.410 | -0.018 |

**中文 A/B 完全一致**（`search_hybrid` 对 zh 透传，验证通过）。

诊断（`diag_hybrid_report.json`）：
- D1 融合生效（`fusion_swallowed=0`），但 **50/100 题两路完全一致**。
- D3 **BM25 单独 R@20 = 0.46，仅 1.93 倍随机基线**（随机 0.238）→ 排序能力弱。
- D4 概括型占 34%，数值型受 PDF 提取拖累（`rq=rv=8` token 数为 0，见 C34）。
- D5 McNemar 不一致对 n=4（2 救 2 推），**样本不足，不输出 p 值**；R@5 差值 95% CI `[-0.08, 0.06]` **含 0**，差异是噪声。

**结论：BM25 在单文档场景前提不满足（IDF 失效、两路同质、擅长题型稀缺），-0.02 落在 1 个标准误以内（bootstrap 均值 -0.01，95% CI `[-0.08, 0.06]`），属噪声，回滚。**

### 设计决策

| # | 决策 | 原因 | 放弃的替代及代价 |
|---|---|---|---|
| H1 | BM25 状态用模块级全局变量，不用 `lru_cache` | 重建语义直白 | `lru_cache`：`cache_clear` 语义绕 |
| H2 | `_is_zh` 内联正则，不 import qa | 避免循环依赖 | import qa：循环 import |
| H3 | `_rrf_fuse` 用 dict 排名映射 | 缺失路 O(1) 判断 | list + `index()`：O(n²) |
| H4 | `bm25_search` 返回 `list[Chunk]` 不带分数 | RRF 只看排名，带分数徒增误用 | 带 BM25 原始分数：下游可能误当余弦 |
| H5 | `search_hybrid` 对 zh 透传 | 中文 BM25 是噪声，透传保证与现状一致 | 中文也 RRF：可能劣化 |
| H6 | `qa.py` 只改 1 行 import 别名 | 签名不变，其余零改动 | 改多处调用点：易漏 |
| H7 | 从 Chroma 读建表 | 数据源唯一 | 从 PDF 重解析：可能不一致 |
| H8 | 空 corpus 降级 None | 契约 8b 不报错 | 建空 BM25Okapi：库内报错 |
| H9 | 测试拆纯逻辑 + 集成两组 | 纯逻辑可进 CI | 全集成：CI 跑不动 |
| 回滚 | 保留 `hybrid_search.py`，`qa.py` 改回纯向量 | 代码无误，多文档场景储备 | 删除：丢失储备 |

### 变更记录

| # | 变更 | 原因 |
|---|---|---|
| C35 | 阶段 2 回滚：`qa.py` 改回 `index_store.search`，`app.py` `_on_upload` 无 `rebuild_bm25`；`hybrid_search.py` / `test_hybrid.py` / `evaluate_hybrid.py` / `diag_hybrid.py` / `rank_bm25` 保留 | 实测单文档场景无净提升（见上表），BM25 前提不满足。保留代码作多文档储备，接入回滚以降复杂度。 |
| C36 | 入口级测试 `test_on_upload_triggers_rebuild` 未纳入最终测试集 | 该测试被 `bm25_search` 懒加载兜底（`_on_upload` 不调 rebuild 时，懒加载也会读到新 Chroma），无法验证契约 2.4 的配对纪律。回滚后 `_on_upload` 不再 rebuild，测试删除，`test_hybrid.py` 回到 11 项。 |
| C37 | 阶段 2a 契约 2.4 的 `_on_upload` rebuild 配对在生产入口未生效 | `app.py` 该调用始终为注释状态；`test_rebuild_sync` 是自包含测试，未覆盖生产入口。回滚后此缺口不再有影响，记录备查。 |
| C38 | 阶段 3 流式：`_stream_llm` 的 `_call()` 标注 `-> Any`，`resp: Any` | zhipuai 类型 stub 把 `stream=True` 的返回错误推断为 `tuple[str, Any]`，运行时实际是 `Stream[ChatCompletionChunk]`（有 `.choices`）。Pylance 报 `chunk.choices` 未知属性。与 C10/C18 同源（stub 未按 stream 参数重载），运行时行为经 8 passed 验证不变。 |
| C39 | 阶段 3 实测：首字延迟主要来自重排（纯 CPU 推理、十几秒量级、随机器变化），瓶颈在重排不在 LLM；新增占位提示 | 流式只优化了 LLM 阶段，但重排在 LLM 之前串行阻塞，用户点发送后长时间无任何反馈。修复：`_on_send` 立即 yield "正在检索并重排…"占位，真 token 到来时清空。19e 口径从"秒级出首字"改为"重排完成后秒级出首字"。重排耗时优化记入后续阶段（需改 rerank.py 或换模型，超出阶段 3 范围）。 |
| C40 | 阶段 4 降级与容错：`degraded` 事件 + `_error_to_message` + pending_prefix | ① `answer_stream` 新增第四种事件 `degraded{reason, fallback_count}`：rerank 抛异常时降级用粗排 top final_k 继续，不抛异常（可预期失败用事件表达，真异常仍抛）；`fallback_count = min(final_k, len(coarse_results))`，degraded 必为首事件、与 refused 互斥、之后仍发 citations。② D1 跳过阈值判断（C3 取舍：粗排余弦分与重排分不同量纲），qa.py 留注释、README 留痕。③ D5 防御分支：rerank 返回空 → refused(empty)。④ `app.py` 新增 `_error_to_message(exc)`：异常→友好文案（Key 缺失/限流/Chroma/兜底），靠关键字识别不引入新异常类型。⑤ `pending_prefix` 机制：degraded 不即时渲染，首 token 到达时拼接，避免空消息闪烁。⑥ 流中途异常永远 append 不重置，覆盖 D1+D2、D1+D3 叠加。⑦ 未知事件类型 print 到 stderr 不崩（向前兼容）。⑧ `answer()` 消费 stream 遇 degraded 静默忽略（非流式调用方对降级透明）。验收 20a–20g 全通过（15 passed）。 |
| C41 | 阶段 5 结构化日志：loguru 双 sink + 汇总日志 + token 用量 + stream_options 探测 | ① 新增 `_setup_logging()`：控制台人类可读（`HH:mm:ss \| LEVEL \| message`）+ 文件 JSON（`logs/app.log`，`serialize=True`，按天轮转留 7 天）；幂等，重复调用不重复加 sink。② 新增 `_log_query()`：一条/请求汇总日志，字段含 `lang/query_len/retrieval_ms/rerank_ms/llm_ms/total_ms/n_candidates/top_score/refused/refuse_reason/degraded/degrade_reason/prompt_tokens/completion_tokens/total_tokens`；用 `logger.bind(**fields).info()` 让字段进 JSON 的 `record.extra`。③ `_stream_llm` 的 yield 从 `str` 改为 `tuple[str, Any]`（`"token"`/`"usage"`），`answer_stream` 消费元组、收集 usage、流结束打日志；`answer_stream` 对外 yield 协议不变（`app.py` 零改动），`RagAnswer.usage` 保持 None（D5）。④ 新增 `_supports_stream_options()`：每次请求用 `inspect` 探测 zhipuai `create()` 是否接受 `stream_options`（结果未缓存，单次 inspect 开销可忽略）。zhipuai 2.1.5 不支持，直接传会抛 `TypeError`；原先"试错→降级"导致每次请求多打一次被拒请求 + WARNING，改为探测后不支持就直接不带。运行时降级分支只匹配明确的 `stream_options`/`include_usage` 关键词（去掉过宽的 `"unknown"`）。⑤ 删除 5 处 `print("[性能日志]...")`，信息并入汇总日志。⑥ `app.py` 的 `_error_to_message` 调用处加 `logger.error`（结构化：`error_type`/`error_msg`/`user_msg`），未知事件告警保留 `print(stderr)`（契约 5e）。⑦ `.gitignore` 加 `logs/`。验收 22a–22j 全通过（15 passed + 真实运行日志验证：中英文 token 用量、拒答 null、隐私无原文）。实测 `_supports_stream_options()` 返回 **False**（`zhipuai 2.1.5.20250825` 的 `Completions.create()` 签名 22 个参数里无 `stream_options`），故不传该参数；token 统计仍可用——SDK 末 chunk 默认带 usage。日志实测中英文各有一条完整记录：prompt 1680–1692、completion 398–976，`total = prompt + completion` 成立。 |
| C42 | 阶段 7 CI/CD 流水线：单 Job 串行 + 模型缓存 + 触发部署 | ① 新建 `.github/workflows/ci.yml`：`push` 主干触发，拆分为 test 和 deploy 两个 Job。② 测试选型：跑 `test_ingest`、`test_index_store`、纯逻辑评估测试，排除真实 API 测试。③ 模型缓存：配合 `snapshot_download` 多线程下载并缓存 `./models`。④ 同步（D4）：放弃第三方 Action，改为原生 `git fetch` + `git reset --hard` + 文件覆盖拷贝，彻底解决远端分支保护（protected branch）和 LFS 残留导致的强推/合并失败。⑤ 部署（D5）：利用 `echo` 追加注释打破 Docker 构建缓存，通过 `sed` 清理乱码防止 `pip` 解析失败，由魔搭平台自动触发 Dockerfile 重建；构建结果由 `deploy` Job 轮询校验（见 C45）。 |
| C43 | 阶段 7 未来演进文档：README 新增「CI/CD 与未来演进」一节 | 总结单人向多人协作演进的方法。① 触发层：引入 PR 和 Branch Protection。② 部署层：引入 GitHub Environments 和人工 Approve。③ 权限层：Secrets 迁移到 Environment 级别，引入 CODEOWNERS。明确记录演进代价：审批增加延迟、分支保护变严、Secrets 迁移有成本。 |
| C45 | 部署门禁：`deploy` Job 轮询空间状态，构建或启动失败即 CI 红 | ① 现象：曾出现「GitHub CI 全绿、魔搭侧构建失败」（2026-11-22 11:34 状态 `BuildImageFailed`），CI 只负责 push，构建结果无人校验。② 做法：push 后用公开只读接口 `https://www.modelscope.cn/api/v1/studio/{repo}` 读 `Data.Status` 与 `Data.DeployedTime`；判成功需 `Status=Running` **且** `DeployedTime` 晚于本次 push 时间（否则旧实例的 `Running` 会被误判成新版本已上线），出现 `*Failed*` 或 30 分钟未上线则 exit 1。③ 另调 `POST /openapi/v1/studios/{repo}/deploy`（Bearer token，实测 HTTP 200）作显式触发，失败不阻断——`git push` 本身就会触发构建（构建日志 `CloneRepoList` 记到的提交正是 CI 写入的 `chore: trigger modelscope rebuild`，即为反证）。④ 踩坑：首版按 `d['Data']` 取值，而 openapi 的返回壳是小写 `data/status`，导致 40 轮全部 `PARSE_ERR`；拉日志的 curl 又没设 `--max-time`，把 job 拖到 45 分钟上限。改为大小写双兼容 + 所有请求限时。⑤ 并发去重：workflow 级 `concurrency: group=ci-${ref}` + `cancel-in-progress: true`——两个 active run 同时向魔搭推送会把构建排成两个，实测耗时明显拉长；新 push 应直接取消旧 run。⑥ 轮询预算：2026-11-22 曾用「60 轮 × 30 秒 ≈ 30 分钟」，当次平台构建耗时 33 分钟，循环在镜像上线前 15 秒放弃，CI 判红而实际已 `Running`、空间 HTTP 200（假阴性）。改为绝对截止时间 60 分钟，deploy job 上限提到 75 分钟；判据本身（`Running` + `DeployedTime` 晚于 push）未变——问题在预算，不在解析或接口。 |
| C46 | 隐私口径收敛：日志**任何级别都不记 query 原文**，不提供 DEBUG 开关 | ① 原状：`_log_query` 末尾有一段 `if logger.level("DEBUG").no >= logger.level("INFO").no: logger.bind(query=query).debug(...)`，注释称"仅 DEBUG 时才记原文"。② 三层问题：判据是两个级别常量相比（10 >= 20）恒为假，与运行环境无关；即使改成 `<=`，`_setup_logging()` 的两个 sink 都写死 `level="INFO"`，DEBUG 仍不落盘；全项目没有任何 `LOG_LEVEL` 环境变量，"开 DEBUG"这个动作不存在。隐私当时靠两层巧合守住，而注释是个"改一行就能看到原文"的陷阱。③ 处置：整段删除，契约 5.3 改为"任何级别都不记原文"。理由——日志会被上传/共享，泄漏不可回收；排查靠结构化字段（`lang`/各段耗时/`top_score`/`refused`）定位问题类型，必要时让用户复述。④ 不采用"配置驱动开 DEBUG"：那等于把一条随时可能误开的泄漏路径留在代码里，收益（本机复现原文）远小于风险。⑤ 验证：grep 全仓 `query_text` 无残留；跑一轮真实请求后 `Select-String -Path logs/app.log -Pattern "<问题原文>"` 仍无输出。 |
| C47 | 评估数据体系：一次遍历产出「逐题明细 + 双阈值报告 + K 两表」 | ① 前置问题：`eval_report.json`（zh 0.18）与 `eval_report_v1.json`（zh 0.15）非同源，且旧报告由未提交的 `evaluate.py` 生成；全仓无逐题 rank+score 明细，K=4 vs K=5 无法拆开讲。② `evaluate.py` 新增 `--report` / `--details` / `--zh-threshold` / `--en-threshold`，`compute_metrics` 额外收集 `_details`（`qid/lang/gold_page/coarse_rank/coarse_gold_score/coarse_top1_score/fine_rank/fine_gold_score/fine_top1_score/threshold/refused`），由 `main()` pop 后单独落盘 `eval_details.json`（140 条），不污染报告字段。③ 召回类指标与阈值无关、拒答类只依赖每题 top1 分数，故 zh 0.15/0.18 由同一份明细离线重算为 `eval_report_zh015.json` / `eval_report_zh018.json`，**不重跑推理**；派生出的 0.15 与 2026-11-07 快照逐条吻合（0.65/0.32/0.375），反证两份报告本就同源。④ `diag_ranks.py` 从"自己再跑 140 题"改为读明细的离线分析器，出四张表：排名分段、分数分布（top1 与 gold 的均值/分位数）、区间迁移、阈值敏感度（0.15/0.18/0.30 三档）。⑤ K 结论：Top-5 的增量是 EN 精排 0 题、ZH 1 题；且 6-10 段之后 gold 分数中位已跌到 0.042–0.181、11-20 段 EN 仅 0.003，这些"命中"在生成阶段不可用 → `final_k=4` 有两条独立证据，不再只靠直方图。⑥ `eval_report_v1.json` 保留原名，新增 `_note` 键注明"历史快照（旧口径 6 指标），正式对比见 zh015/zh018"。⑦ 入库：`test_final.py` 加 `__main__` 保护（pytest 收集 0 项、不重写权重/不重建索引），`diag_ranks.py`、`evaluate.py` 一并提交，消除"README 目录树列了仓库里没有的文件"的不一致。 |
