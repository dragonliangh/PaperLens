---
domain: 
tags: 
datasets: 
  evaluation:
  test:
  train:
models: 
license: Apache-2.0
---

# PaperLens —— 学术论文 RAG 助手

上传学术论文 PDF，基于原文回答技术问题，并给出**页码级引用**；检索不到证据时**明确拒答**，而不是强行生成。

> 当前进度：第 1–8 块 + 阶段 1–7 已实现。测试现状（2026-11-21 本机全量 pytest）：**37 passed / 8 failed / 3 skipped**，失败集中在 `test_qa.py` 中 6 个标注"不需 Key"的用例（详见「本地运行测试」）。已实现范围包括：解析分块 → 向量索引 → 检索生成引用 → Gradio 界面 → Docker 部署 → 两阶段检索 → 评估与拒答 → 对照组评估 → 消融 → 混合检索负结果 → 流式 → 降级容错 → 结构化日志 → 架构图 → CI/CD。
>
> 已知偏差：契约 2.4（上传即重建混合索引）未在生产入口生效——混合检索实测无增量后已回滚为纯向量，`_on_upload` 只做解析与建索引（见 DESIGN C37、`app.py:94-107`）。

---

## 目录结构

```text
PaperLens/
├── data/
│   ├── sample.pdf              # 样例论文（LoRA），用于测试与演示
│   └── chroma/                 # Chroma 持久化目录（第 2 块自动生成）
├── models/                     # 不进 Git（.gitignore），需自行下载，见「离线模型」
│   ├── bge-small-en-v1.5/      # 第 1/2 块：嵌入模型
│   └── bge-reranker-base/      # 第 6 块：重排模型
├── logs/
│   └── app.log                 # 结构化日志（运行时生成，已 gitignore）
├── .dockerignore               # 构建上下文排除（含 models/）
├── .gitignore                  # 排除 models/、logs/、INTERVIEW.md
├── .github/workflows/ci.yml    # 阶段 7：测试 + 部署流水线
├── ingest.py                   # 第 1 块：PDF 解析 + 分块
├── test_ingest.py              # 第 1 块验收测试（5 passed）
├── index_store.py              # 第 2 块：嵌入 + 向量库索引
├── test_index_store.py         # 第 2 块验收测试（7 passed）
├── qa.py                       # 第 3/6/7 块：检索 + 重排 + 生成 + 引用 + 拒答 + 流式 + 降级
├── test_qa.py                  # 第 3 块 + 阶段 3/4 验收测试（15 个用例；设 Key 时 15 passed，无 Key 实测 6 passed / 6 failed / 3 skipped）
├── rerank.py                   # 第 6 块：Cross-Encoder 精排
├── app.py                      # 第 4 块：Gradio 界面（上传建索引 + 流式 + 降级）
├── evaluate.py                 # 第 7 块：主评估脚本（140 题）
├── evaluate_ctrl.py            # 第 7 块补充：对照组评估（20 题 × 2 语言）
├── evaluate_ablation.py        # 阶段 1：消融实验（有/无重排）
├── evaluate_hybrid.py          # 阶段 2b：混合检索 A/B 评估
├── test_eval.py                # 第 7 块验收测试
├── test_hybrid.py              # 阶段 2：混合检索验收测试（11 passed）
├── hybrid_search.py            # 阶段 2：混合检索（BM25+向量+RRF，已回滚接入，保留作储备）
├── eval_set.json               # 第 7 块：140 条主评估集
├── ctrl_eval_set.json          # 第 7 块补充：40 条对照组评估集
├── thresholds.json             # 第 7 块：按语言的拒答阈值
├── eval_report.json            # 第 7 块：主评估输出（自动生成；zh=0.18，含 fine_rank_histogram）
├── eval_details.json           # 第 7 块：逐题明细（140 条，含粗排/精排 rank 与 gold/top1 分数；分段表与阈值对比的数据底座）
├── eval_report_zh015.json      # zh 阈值 0.15 的正式报告（由同一次遍历的明细离线重算，与 zh018 可直接对比）
├── eval_report_zh018.json      # zh 阈值 0.18 的正式报告（与 eval_report.json 同数，单列便于对比引用）
├── eval_report_v1.json         # 历史快照（2026-11-07，zh 阈值 0.15，旧口径 6 个指标；文件内 _note 已注明用途）
├── ctrl_eval_report.json       # 第 7 块补充：对照组评估输出（自动生成）
├── ablation_report.json        # 阶段 1：消融报告（自动生成）
├── hybrid_eval_report.json     # 阶段 2b：混合检索评估输出（自动生成）
├── diag_hybrid_report.json     # 阶段 2b：混合检索诊断输出（D1–D5）
├── diagnose.py                 # 诊断脚本：失败样本
├── diagnose_pushed_away.py     # 诊断脚本：精排推走的题
├── diagnose_remaining.py       # 诊断脚本：剩余未命中题
├── diagnose_threshold.py       # 诊断脚本：阈值扫描
├── diag_ranks.py               # 诊断脚本：读 eval_details.json，出排名分段 / 分数分布 / 区间迁移 / 阈值敏感度四张表（不再自己重跑）
├── diag_hybrid.py              # 诊断脚本：混合检索
├── download_model.py           # 单模型下载（ModelScope）
├── download_full_model.py      # 完整模型目录下载
├── review_candidates.py        # 生成人工复核候选
├── review_candidates.txt       # 候选清单（含 PDF 解析噪声，仅供人工参考）
├── test_final.py               # 手动修复脚本（从 .bin 重生成 .safetensors）+ 端到端自检；已加 __main__ 保护：pytest 收集到 0 项、不会重写权重或重建索引
├── Dockerfile                  # 第 5 块：Docker 构建文件
├── docker.yaml                 # 魔搭创空间配置
├── ms_deploy.json              # 魔搭 Docker 部署配置
├── requirements.txt            # 依赖版本（不含 torch / pytest，见「依赖」）
├── README.md                   # 本文件
├── README1.md                  # 魔搭创空间门面页（指向本文件）
└── DESIGN.md                   # 契约、设计决策、变更记录
```

---

## 架构图

系统的数据流与分层结构如下（按请求执行顺序自上而下排列）：

```text
┌────────────────────────────────────────────────────────┐
│ 1. 用户边界                                             │
│   上传 PDF → 提问 → 看答案 + 引用                         │
└───────────────────────┬────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ 2. 界面层 (app.py)                                      │
│   [Gradio UI] → [_on_upload 上传] → [解析 → 建索引]      │
│            └──→ [_on_send 流式] → [_error_to_message]   │
│                       ↑                                │
│                [pending_prefix 降级前缀]                │
└───────────────────────┬────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ 3. 业务层 (qa.py)                                       │
│   [answer_stream 流式] → [search 粗排 top-20]            │
│           ↓                   ↓                        │
│   [rerank 精排 top-4] ← [拒答阈值 en=0.3 / zh=0.18]      │
│           ↓                                            │
│   [拼 prompt] → [调 GLM 生成] → [解析引用]               │
│   事件 4 种：token / citations / refused / degraded      │
└───────────────────────┬────────────────────────────────┘
                        ↓
┌────────────────────────────────────────────────────────┐
│ 4. 存储 / 模型层                                         │
│   [Chroma 向量库]  [BGE-small 嵌入]                     │
│   [BGE-reranker 重排] [GLM-4.7-Flash 生成]              │
│   [logs/app.log 结构化日志]                             │
└────────────────────────────────────────────────────────┘
```
---

## 环境要求

| 场景 | 系统 | 说明 |
|---|---|---|
| 本地开发 | Windows 10 / 11 + PowerShell | 本项目在 Win10 + Python 3.13 上开发 |
| 容器 / 云端 | Linux（`python:3.13-slim`） | `Dockerfile` 与 CI 均跑 Linux；命令见下方 Linux 栏 |

Python 3.13。

### 依赖

`requirements.txt` 锁定全部项目依赖，每行都带精确版本号（Python 3.13.0 实测通过）：`pdfplumber==0.11.8`、`Pillow==11.3.0`、`transformers==5.22.1`、`sentence-transformers==6.2.0`、`chromadb==1.5.9`、`zhipuai==2.1.5.20250825`、`sniffio==1.3.1`、`gradio==5.49.1`、`rank_bm25==0.2.2`、`loguru==0.8.0`、`modelscope==1.40.2`。`pdfplumber` 锁 0.11.8 而不是 0.11.10，因为 0.11.10 要求 `Pillow>=12.2.0`，与 `gradio 5.49.1` 的 `Pillow<12.0` 无交集（详见「已知限制 第 5 块」）。

**`torch==2.15.0+cpu` 与 `pytest==9.2.0` 不在 `requirements.txt` 里**，必须单独安装（见下第 2、4 步），版本要与这里保持一致。

### 首次配置（按顺序执行）

0. 获取代码并创建虚拟环境：

   ```powershell
   git clone https://www.modelscope.cn/studios/oceanlh/PaperLens-RAG.git PaperLens
   cd PaperLens
   python -m venv .venv
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
   .\.venv\Scripts\Activate.ps1
   ```

   Linux / 容器内：

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

1. 配置 pip 国内镜像（只需一次，可选）：

   ```powershell
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```

2. 安装 CPU 版 torch（**单独装，不要用默认源**，否则会拉入 CUDA 依赖）：

   ```powershell
   pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
   ```

   > 若该版本在你的 Python 上不可用，按 PyTorch 官网 CPU 源选择可用版本（本机已验证：Python 3.13.0 下 `torch 2.15.0+cpu` 可安装并正常运行）。

3. 安装其余依赖（**不要逐个裸装包名**，否则会绕过锁定版本，装出 6.x 的 Gradio 或触发 Pillow 冲突）：

   ```powershell
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. 安装测试框架：

   ```powershell
   pip install pytest==9.2.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

5. 下载两个模型到 `models/`（**这一步不能跳过**：`models/` 已被 `.gitignore` 排除，新克隆的仓库里没有模型，而 `ingest.py` / `index_store.py` / `rerank.py` 会强制离线加载）：

   ```powershell
   python -c "from modelscope import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5', local_dir='./models/bge-small-en-v1.5'); snapshot_download('BAAI/bge-reranker-base', local_dir='./models/bge-reranker-base')"
   ```

   预期：`models/bge-small-en-v1.5/` 与 `models/bge-reranker-base/` 两个目录出现，含 `config.json`、tokenizer 文件与权重。

   无法联网的机器：把这两个目录从已有环境整体拷贝到 `models/` 下，效果等价。

6. 设置智谱 API Key（第 3/6/7 块与问答需要）：

   ```powershell
   $env:ZHIPU_API_KEY = "你的key"
   ```

   验证：`echo $env:ZHIPU_API_KEY` 能看到值。不要把 Key 写进代码或提交到仓库。

### 预期输出与失败处置

| 现象 | 原因 | 处理 |
|---|---|---|
| `ModuleNotFoundError: No module named 'modelscope'` | 没装 `requirements.txt`，或只按旧版命令裸装了部分包 | 执行第 3 步 |
| 启动时打印「模型下载失败」/ 找不到 `models/bge-small-en-v1.5` | 未执行第 5 步，或离线加载但目录不存在 | 补下载或从别处拷贝 `models/` |
| `OSError: [Errno 98] Address already in use`（或 Windows 的 10048） | 7860 被占用 | 结束占用进程，或改 `app.py` 里的 `server_port` |
| `ValueError: ZHIPU_API_KEY 未设置。请在终端执行 $env:ZHIPU_API_KEY = "你的key" 后重试。` | 未设 Key 或换了终端窗口 | 重设环境变量（第 6 步） |
| 上传 PDF 后提问得到「根据当前论文库，我无法回答这个问题。」（检索为空或低于阈值时的拒答文案） | 索引为空（建索引失败或未上传） | 看控制台日志；重跑「本地建索引」 |
| 界面聊天框不显示内容 | Gradio / Pillow 版本不匹配（不是 `gradio 5.49.1` + `Pillow 11.3.0` 这组） | `pip show gradio Pillow` 核对，按 `requirements.txt` 重装 |

### 离线模型

- **本地推理离线**：`ingest.py`、`index_store.py`、`rerank.py` 在导入时设置 `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`，只读 `./models/` 下的本地目录。
- **但首次启动会尝试联网**：`app.py` 在 `__main__` 里无条件调用 ModelScope `snapshot_download` 校验并补全模型（`app.py:271-282`）；启动时会重新下载并覆盖已有权重（实测会被重下），完全离线的机器会卡在这一步。
- `models/` 不在 Git 仓库里（`.gitignore` 第 1 行），克隆后必须自行准备，或用上面的 `snapshot_download`。

---

## 运行

### 本地运行测试

测试的前置条件（缺一项就会失败或跳过）：

| 测试文件 | 前置 |
|---|---|
| `test_ingest.py` | `models/bge-small-en-v1.5/` 就位、`data/sample.pdf` 存在 |
| `test_index_store.py` | 同上；测试会写/清 `data/chroma`，跑完索引为空 |
| `test_qa.py` | 已建索引；9 例需要 `$env:ZHIPU_API_KEY`（9a/9b/19a 会自动跳过）；20a–20g 只 mock 了 `rerank`/`_stream_llm`、未 mock `_get_client()`，无 Key 时 6 例均失败（5 例 ValueError，20d 为 AssertionError） |
| `test_eval.py` | 已建索引；会调用真实 API（消耗额度） |
| `test_hybrid.py` | 已建索引；`hybrid_search.py` 保留，测试仍有效 |

```powershell
pytest test_ingest.py test_index_store.py test_qa.py -v
```

预期：`test_ingest` 5 passed、`test_index_store` 7 passed；`test_qa` 设 Key 时 15 passed，无 Key 实测 6 passed / 6 failed / 3 skipped（2026-11-21）。

```powershell
pytest test_eval.py -v
```

预期：10 个用例，其中 2 例走真实 API（需设 Key）；无 Key 实测 8 passed / 2 failed。

```powershell
pytest test_hybrid.py -v
```

预期：11 passed。

### 本地建索引

```powershell
python -c "from ingest import parse_and_chunk; from index_store import build_index; build_index(parse_and_chunk('data/sample.pdf')); print('ok')"
```

### 本地启动 Web 界面

```powershell
python app.py
```

启动时会**先清空 `data/chroma` 里的旧索引，再用 `data/sample.pdf` 重建一次**（`app.py:284-297`），然后预热重排模型。所以手动建好的索引会被重置；要换论文，直接在界面里上传即可。

浏览器打开控制台输出的本地地址（通常是 `http://127.0.0.1:7860`）。云端（设置 `MODELSCOPE_ENVIRONMENT`）会监听 `0.0.0.0:7860`。

### 容器运行（Docker）

```powershell
# 构建（在项目根目录执行；.dockerignore 已排除 .venv / models/ / data/chroma）
docker build -t paperlens:0.1 .

# 运行：Key 用 -e 注入，端口对齐 7860
docker run --rm -p 7860:7860 -e ZHIPU_API_KEY="你的key" paperlens:0.1
```

注意：`models/` 被 `.dockerignore` 排除，**镜像内不含模型**，容器首次启动会由 `app.py` 从 ModelScope 下载；离线环境需先把 `models/` 挂载进容器：

```powershell
docker run --rm -p 7860:7860 -e ZHIPU_API_KEY="你的key" -v "${PWD}/models:/app/models" paperlens:0.1
```

魔搭创空间侧不需要手动执行这些命令：`docker.yaml` 指定 Dockerfile，`ms_deploy.json` 声明 `sdk=docker` 与 `app_port=7860`，平台按仓库自动构建，端口与 `Dockerfile` 的 `EXPOSE 7860`、`app.py` 的 `launch` 三处一致。

### 代码调用示例

```powershell
python -c "from qa import answer; r = answer('How much does LoRA reduce trainable parameters?'); print(r.answer); [print(f'[{i+1}] {c.doc_name} p{c.page_start}-{c.page_end} | {c.chunk_id}') for i, c in enumerate(r.citations)]"
```

返回 `RagAnswer`，字段与 `Citation` 定义见 `qa.py:301-337`：

| 对象 | 字段 | 类型 |
|---|---|---|
| `RagAnswer` | `answer` | `str` |
| | `citations` | `list[Citation]` |
| | `usage` | `dict \| None`（当前恒为 `None`，token 只进日志，见 DESIGN C41③） |
| `Citation` | `chunk_id` / `doc_name` / `page_start` / `page_end` / `text_snippet` | `str` / `str` / `int` / `int` / `str` |

流式入口为 `answer_stream(query, top_k=20, final_k=4)`，产出四种事件：`token` / `citations` / `refused` / `degraded`（`qa.py:611-624`）。

### 跑评估

评估脚本只调用 `search` 与 `rerank`，**不调 GLM、不消耗额度**，且要求已建索引。**全程 CPU 推理、未用 GPU**：本机 4 核 8 线程、140 题一次全量遍历在十几到二十几分钟量级（脚本每 5 题打印一次进度，不会长时间无反馈）。

```powershell
# 主评估：写 eval_report.json，并落盘逐题明细（分段表与阈值对比的数据底座）
python evaluate.py --details eval_details.json

# 由明细出四张表：排名分段 / 分数分布 / 区间迁移 / 阈值敏感度（纯离线，秒级）
python diag_ranks.py --details eval_details.json --out 表.md
```

140 题（70 EN + 70 ZH）。`--report` 可改输出路径，`--zh-threshold` / `--en-threshold` 可临时覆盖阈值。

> 阈值的正式对比**不需要重跑**：召回类指标与阈值无关，拒答类指标只依赖每题的精排 top1 分数，而它已落在 `eval_details.json` 里。因此 zh 0.15 与 0.18 的报告由同一份明细离线重算（`eval_report_zh015.json` / `eval_report_zh018.json`），两条命令即可复现，见「3. 拒答机制评测」。

```powershell
python evaluate_ctrl.py
```

40 条对照组。输出表格 + 写入 `ctrl_eval_report.json`。

```powershell
python evaluate_ablation.py
```

100 题消融（有/无重排）。输出表格 + 写入 `ablation_report.json`。

```powershell
python evaluate_hybrid.py
```

100 题 × 2 组混合检索 A/B。输出表格 + 写入 `hybrid_eval_report.json`。

---

## 已知限制

这些是当前已知的边界与待修项，详见 `DESIGN.md`。其中 `torch.load` patch 与 C22 不符仍属待修 bug；DEBUG 日志分支已于 2026-11-22 整段删除（改为「任何级别都不记原文」，见 C46），不再是待修项。

### 第 1 块：PDF 解析

- PDF 双栏排版可能导致文字顺序错乱；公式、表格抽取不保证正确。
- 单词间空格可能丢失（如 `Edward Hu` → `EdwardHu`）。
- 章节识别不保证成功，`section` 可能为空。

### 第 2 块：向量索引

- `build_index` 假设一批 chunks 属于同一篇论文。
- `score = 1 - cosine_distance` 是相对排序依据，不是概率。
- 纯向量检索对「语义相关但未直接回答」的段落区分度有限，这是 P1 引入重排的直接动机。

### 第 3 块：LLM 生成

- LLM 引用可能错位：机制本身正常，是 LLM 标注不精确。这是 RAG 已知问题，由第 7 块 Faithfulness 评估量化。
- 免费档 `completion_tokens` 可能高于可见答案长度，属平台计费口径，不影响功能。
- **免费模型存在账户级限流**：智谱 `glm-4.7-flash` 在高峰期会返回 HTTP 429，错误码为 `1302`（账户已达速率限制）或 `1305`（模型当前访问量过大）。`qa.py` 已把这两个码列入限流识别集，并在首 token 前等待 `RETRY_DELAY_SECONDS` 后重试；但退避重试仍可能连续被限——2026-11-21 本机对同一问题连续尝试三次（含间隔等待）均为 429，而同一把 Key 的非流式调用可以正常返回。用户侧看到的是「生成服务繁忙，请稍后重试」；这是平台配额与时段负载所致，不是代码缺陷，也不是 Key 失效。

### 第 4 块：Gradio 界面

- 上传的 PDF 存在 Gradio 的临时目录，进程重启即失效。
- 索引是全局的：多个用户上传不同论文会共存于同一 Collection，检索时跨论文混合。
- **流式输出已做（阶段 3），但只优化 LLM 阶段**：答案逐字出，首字延迟仍有十秒量级（全程 CPU 推理、未用 GPU，耗时随机器与负载变化），瓶颈在重排而非 LLM——`logs/app.log`（2026-11-18，7 条完整请求）显示重排占掉首字延迟的绝大部分。点发送后立即显示"正在检索并重排…"占位缓解白屏；重排耗时优化列为后续独立项。
- **占位提示不显示用时/阶段**：用户不知道卡在哪一步、还要等多久。分阶段文案和计时记为后续优化项（S8）。

### 第 5 块：Docker 部署（魔搭）

- **依赖版本冲突**：`pdfplumber 0.11.10` 要求 `Pillow>=12.2.0`，而 `gradio 5.49.1` 要求 `Pillow<12.0`——两个约束无交集，这两版**不可能同时装上**（干净的 `pip install` 会直接报依赖解析失败）。`requirements.txt` 的写法是 `pdfplumber==0.11.8` + `Pillow==11.3.0`：0.11.8 只要求 `Pillow>=9.1`，与 gradio 相容，是能装成功的组合。**本机 venv 已于 2026-11-21 按 `requirements.txt` 对齐**（实测 `pdfplumber 0.11.8` + `pdfminer.six 20251107` + `Pillow 11.3.0` + `gradio 5.49.1`）：`pip check` 中 `pdfplumber↔Pillow` 与 `kubernetes↔pydantic` 两条冲突均已消除，现仅剩 `hf-gradio`（不在 `requirements.txt` 内）一条。对齐后重建索引仍为 **84 chunk**，3 个固定 query 的 top-4 检索结果与对齐前逐条一致（仅 1 条分数有 1e-6 浮点噪声），故既有评估结论不受影响。
- **Gradio 版本**：锁定 `gradio 5.49.1` + `Pillow 11.3.0`（`pdfplumber 0.11.8` 要求 `Pillow>=9.1`，`gradio 5.49.1` 要求 `Pillow<12.0`，这是唯一相容组合）；聊天区必须显式声明 `type="messages"`。早期笔记曾把「聊天框不显示内容」归因为「Gradio 6.x 的 iframe 渲染问题」，该说法无实测支撑，已改为版本兼容性口径（见 `DESIGN.md` C15）。
- **镜像体积**：Linux 下默认安装的 `torch` 会附带大量 CUDA 依赖，使镜像显著膨胀。解决办法是从 PyTorch CPU 源单独安装 `torch==2.15.0`，体积随之大幅缩小（仓库未留存构建日志，具体体积不再引用）。
- **性能表现**：云端体验明显快于本机，但未留存云端日志，具体耗时不再引用；本机 `logs/app.log`（2026-11-18）显示检索在数十毫秒级、LLM 在数秒到十几秒级。本机全程 CPU 推理、未用 GPU，耗时随机器与负载变化。前端 Gradio 渲染可能额外增加等待感。
- **平台构建耗时波动、失败原因要人工看**：「构建 + 部署」在本机观测为 6–16 分钟区间（云端队列与缓存命中情况不同），故 `deploy` Job 给 45 分钟上限、轮询窗口 30 分钟。超时或状态出现 `*Failed*` 时 CI 只报 `BUILD_FAILED`，具体失败阶段仍需到空间页「构建日志」查看——CI 内不拉日志（该端点需鉴权、且可能是长轮询，容易把 job 拖住）。

### 第 6 块：两阶段检索与云端修复

- **两阶段检索**：向量粗排（top-20）→ Cross-Encoder 精排（top-4）。曾观察到含答案的摘要页 p1 从粗排第 4 名升至第 1 名（分数 0.76→0.89）；这是单次人工观察，未留存当时的排名明细。
- **重排分数**：Cross-Encoder 输出的分数经 Sigmoid 归一化到 0~1，**与余弦相似度量纲不同**。第 7 块设拒答阈值时必须使用重排分数。
- **CPU 性能**：`bge-reranker-base` 在 CPU 上必须限制 `batch_size=4`，否则内存交换会让单次重排显著变慢（DESIGN:340 记录过该对比，但未留压测日志，故不引用具体耗时）。
- **重排耗时不稳定**：`logs/app.log`（2026-11-18，7 条完整请求）均为十秒量级；另有 1 条数十秒的记录，该次未进入 LLM，疑为冷启动。全程 CPU 推理、未用 GPU，耗时随机器与负载变化。重排是首字延迟的主要来源，优化（换更小 reranker / 减 top_k / 量化）列为后续独立项（S7）。
- **云端模型加载**：曾遭遇两个问题——(1) Git LFS 传输导致 `.safetensors` 文件头损坏；(2) 强制走 `.bin` 又触发 PyTorch 2.6+ 的 `weights_only=True` 安全限制。曾用 `test_final.py` 从 `.bin` 重新生成 `.safetensors` 并推送云端，试图绕开 `torch.load`。
  **当前状态与上述说法不一致**：`ingest.py:14-19` 在导入时仍全局替换 `torch.load` 并强制 `weights_only=False`，仓库里 `models/*/pytorch_model.bin` 也仍在。DESIGN C22「已移除所有 monkey-patch」与代码相反，属待修偏差；在修好之前，以 `ingest.py` 的实际行为为准。

---

## 环境变量

| 变量 | 作用 | 是否必填 |
|---|---|---|
| `ZHIPU_API_KEY` | 智谱 GLM 的 API Key，`qa.py:358` 读取 | 问答必填；`test_eval` 中 2 个真实 API 用例必填。四个评估脚本本身不调 GLM（见「跑评估」），无 Key 也能跑 |
| `MODELSCOPE_ENVIRONMENT` | 非空即视为魔搭云端，`app.py:320-322` 改为监听 `0.0.0.0:7860` | 云端由平台设置，本地不设 |
| `HF_HUB_OFFLINE` / `TRANSFORMERS_OFFLINE` | 强制离线加载本地模型；代码在 `ingest.py:21-22`、`index_store.py:16-17`、`rerank.py:25-26` 内置为 `1` | 不需手动设置 |

## 日志

### 位置与格式

- **控制台**：人类可读，`HH:mm:ss | LEVEL | message`。
- **文件**：`logs/app.log`，每行一条 JSON，按天轮转（`rotation="00:00"`），保留 7 天（`retention="7 days"`）。`logs/` 在 `.gitignore` 里，不进仓库。

### 结构化字段（一条/请求）

每次 `answer_stream` 调用结束时打一条 `query done`，字段：

| 字段 | 说明 |
|---|---|
| `lang` | `"en"` / `"zh"` |
| `query_len` | 问题字符数（任何级别都不记原文，见「隐私」） |
| `retrieval_ms` / `rerank_ms` / `llm_ms` / `total_ms` | 各阶段耗时 |
| `n_candidates` | 粗排返回条数 |
| `top_score` | 重排最高分（降级/拒答时为 null） |
| `refused` / `refuse_reason` | 是否拒答及原因（`"empty"` / `"below_threshold"`） |
| `degraded` / `degrade_reason` | 是否降级及原因（`"rerank_failed"`） |
| `prompt_tokens` / `completion_tokens` / `total_tokens` | token 用量（拒答/降级时为 null） |

`app.py` 的异常另打一条 `query error`，字段 `error_type` / `error_msg` / `user_msg`。

### 隐私

**任何级别都不记问题原文**，只记 `query_len`（问题字符数），也不提供"开 DEBUG 看原文"的开关。

这条是**代码层面的硬约束**：`qa.py` 的 `_log_query()` 里不存在任何写入 `query` 字段的语句，所以不是"默认不记、调试可开"，而是**没有这个能力**。理由：日志文件会被上传/共享，泄漏一次无法回收；排查可以靠结构化字段（`lang` / 各段耗时 / `top_score` / `refused`）先定位问题类型，再让用户复述原问题。

这里修过一处问题（2026-11-22）：原先有一个自己以为"DEBUG 才记原文"的分支 —— `if logger.level("DEBUG").no >= logger.level("INFO").no: logger.bind(query=query).debug(...)`。它有三层问题：① 两个级别常量相比，`10 >= 20` 恒为假，与运行环境无关；② 即使把判据改成 `<=`，`_setup_logging()` 的两个 sink 都写死 `level="INFO"`，DEBUG 仍不落盘；③ 全项目没有任何 `LOG_LEVEL` 环境变量，所谓"开 DEBUG"这个动作根本不存在。也就是说，隐私当时是**靠两层巧合**守住的，而那句"改判据就能看到原文"的注释是个陷阱。最终按"不留可被误修的开关"处理：整段删除，契约 5.3 改成"任何级别都不记原文"（DESIGN C46）。

### token 用量

流式下 usage 在**最后一个 chunk** 里。`qa._supports_stream_options()`（`qa.py:488`）用 `inspect.signature` 探测当前 SDK 的 `create()` 是否接受 `stream_options`，本机实测**返回 False**（zhipuai 2.1.5.20250825 的签名里没有该参数，直接传 kwarg 会抛 TypeError）；函数在请求路径内**每次调用**（`qa.py:559`），并未缓存。

注意：`qa.py:492` 的 docstring 结论是**对的**——「不接受该参数、直接传会抛 TypeError」与实测一致。真正的不准确处是同一段 docstring 里的「改为模块加载时探测一次」，与实际"每次请求调用、无缓存"不符。另需说明：**参数不被接受 ≠ 拿不到 token**——SDK 在流式最后一个 chunk 默认带 usage，故 `logs/app.log`（2026-11-18）7 条完整请求都有 token 数。

### 实测样本

| 问题 | lang | prompt_tokens | completion_tokens | total_tokens |
|---|---|---|---|---|
| 英文正常问答 | en | 1692 | 484 | 2176 |
| 中文正常问答 | zh | 1692 | 536 | 2228 |
| 不可答（拒答） | en | null | null | null |

取自 `logs/app.log`（2026-11-18）的两条完整请求。prompt_tokens 相同（1692）：prompt 里拼的是英文原文，中英文问题只是末尾 query 不同，检索到的 4 段英文原文一样。completion_tokens 中文多 52：中文回答用字更多。

### 已知限制

- **免费档 GLM 高峰期慢**：日志可查 `llm_ms`。本机日志（2026-11-18）最慢一次为数十秒量级（全程 CPU 推理、未用 GPU，耗时随机器与负载变化）；另有一次总耗时更长的说法，但分项未留存，不再引用。
- **重排耗时不稳定**：日志可查 `rerank_ms`，本机 7 条完整请求均在十秒量级，另有 1 条数十秒的记录（未进入 LLM，疑为冷启动），是首字延迟主要来源（S7 后续优化）。

---

## 路线图

> 本表「序号」是路线图顺序，不是 DESIGN 里的块号（DESIGN 共 8 块）；两者不要混用。

| 序号 | 内容 | 状态 |
|---|---|---|
| 1 | PDF 解析 + 分块 | ✅ 完成 |
| 2 | 嵌入 + 向量库索引（Chroma） | ✅ 完成 |
| 3 | 检索 + 生成 + 引用（GLM-4.7-Flash） | ✅ 完成 |
| 4 | Gradio 界面 | ✅ 完成 |
| 5 | Docker 化 + 魔搭创空间部署 | ✅ 完成 |
| 6 | 两阶段检索（重排，P1） | ✅ 完成 |
| 7 | 评估脚本 + 拒答机制（P1） | ✅ 完成 |
| 7+ | 对照组评估（P1 补充） | ✅ 完成 |
| 8 | 消融实验（有/无重排） | ✅ 完成 |
| 9 | 混合检索（BM25+向量+RRF） | ⚠️ 实测无增量，已回滚接入（代码保留） |
| 10 | 流式输出（`answer_stream` + 占位提示） | ✅ 完成 |
| 11 | 降级与容错（5 个降级点 + 友好文案） | ✅ 完成 |
| 12 | 结构化日志（loguru + token 用量） | ✅ 完成 |
| 13 | CI/CD（GitHub Actions 自动测试 + 自动部署魔搭） | ✅ 完成 |
| 14 | 分阶段文案 / 语义缓存 / 查询改写（P2） | ⏸ 以后再说 |

---

## 评估结果

### 评估集设计

- **规模**：140 条（英文 70 + 中文 70，中英文严格语义对应，id 一一对应）。
- **可答题**：每种语言 50 道，`gold_page` 人工标注，命中判定为「`gold_page` 落在 chunk 的 `[page_start, page_end]` 区间内」。
- **不可答题**：每种语言 20 道，按 2:2:1 结构设计——8 道完全无关（unrelated）、8 道半相关（semi_related）、4 道反事实（counterfactual）。
- **中文题处理**：中文题与英文题语义严格对应，`gold_page` 和 `gold_keywords` 与英文完全一致（keywords 用英文原词，因为检索池是英文原文）。

### 1. 同 K 截断对比（消融实验）
为了公平评估 Cross-Encoder 的真实贡献，我们在相同的上下文窗口（Top-K）限制下进行对比：

| 截断口径 | 指标 | 粗排 (纯向量) EN/ZH | 精排 (Cross-Encoder) EN/ZH | 收益 (Δ) EN/ZH |
| :--- | :--- | :--- | :--- | :--- |
| **K=4 (生产口径)** | **Recall** | 0.54 / 0.22 | **0.64 / 0.50** | **+0.10 / +0.28** |
| | **MRR** | 0.35 / 0.12 | **0.43 / 0.40** | +0.08 / +0.28 |
| **K=5 (对照口径)** | **Recall** | 0.60 / 0.24 | 0.64 / 0.52 | +0.04 / +0.28 |
| | **MRR** | 0.36 / 0.13 | **0.43 / 0.40** | +0.07 / +0.27 |
| **K=20 (天花板)** | **MRR** | 0.39 / 0.18 | **0.45 / 0.43** | +0.06 / +0.25 |

*现象解读：中文由于 BGE-small-en 单语模型的跨语言劣势，粗排极弱，重排将其拉升了一倍以上。而英文在 K=4 到 K=5 时，精排 Recall 维持在 0.64，说明对于英文，Top-4 是完美的召回饱和点。*

### 2. 为什么选择 `final_k=4`？（排名分段 + 分数分布）

两张表都来自**同一次**全量遍历的逐题明细（`eval_details.json`，2026-11-22 22:24），可由 `eval_report.json` 的 `coarse_rank_histogram` / `fine_rank_histogram` 与 `diag_ranks.py` 直接复算——不再是"早期一次性诊断、报告里查不到"的数字。

**表 A　排名分段**（可答题，命中 gold 的第一名次落在哪一段；括号内为占可答题比例）

| 排名区间 | EN 粗排 | EN 精排 | ZH 粗排 | ZH 精排 |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1** | 11 (22%) | 15 (30%) | 3 (6%) | 16 (32%) |
| Top-2 | 7 (14%) | 6 (12%) | 3 (6%) | 6 (12%) |
| Top-3 | 6 (12%) | 8 (16%) | 5 (10%) | 2 (4%) |
| Top-4 | 3 (6%) | 3 (6%) | 0 | 1 (2%) |
| **Top-4 合计** | **27 (54%)** | **32 (64%)** | **11 (22%)** | **25 (50%)** |
| Top-5 | 3 (6%) | **0** | 1 (2%) | 1 (2%) |
| 6-10 | 7 (14%) | 6 (12%) | 14 (28%) | 9 (18%) |
| 11-20 | 7 (14%) | 6 (12%) | 11 (22%) | 2 (4%) |
| miss | 6 (12%) | 6 (12%) | 13 (26%) | 13 (26%) |

**表 B　分数分布**（按精排分段；`top1` = 该题精排第 1 名的分数，也就是决定拒答与否的那个数；`gold` = 命中 chunk 自身的分数，中位数）

| 精排区间 | EN 题数 | EN top1 中位 | EN gold 中位 | ZH 题数 | ZH top1 中位 | ZH gold 中位 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Top-1 | 15 | 0.978 | 0.978 | 16 | 0.454 | 0.454 |
| Top-2 | 6 | 0.375 | 0.311 | 6 | 0.523 | 0.109 |
| Top-3 | 8 | 0.822 | 0.425 | 2 | 0.594 | 0.480 |
| Top-4 | 3 | 0.931 | 0.277 | 1 | 0.048 | 0.008 |
| Top-5 | 0 | — | — | 1 | 0.646 | 0.100 |
| 6-10 | 6 | 0.929 | 0.181 | 9 | 0.337 | 0.042 |
| 11-20 | 6 | 0.546 | **0.003** | 2 | 0.494 | 0.060 |
| miss | 6 | 0.499 | — | 13 | 0.467 | — |

**数据解读：**
1. **第 5 个坑位值不值**：K=4→5，英文精排命中数 `32 → 32`（**0 收益**），中文 `25 → 26`（**+1 题**）；粗排口径英文 `27 → 30`（+3）、中文 `11 → 12`（+1）。也就是说 K=5 只买到中文那 1 道题，代价是上下文多 25% 的 token。
2. **第 5 名之后的"命中"多半是分数噪声**：表 B 显示，6-10 段的 gold 中位分数已掉到英文 0.181 / 中文 0.042，11-20 段更是英文 **0.003** / 中文 0.060 —— 这些 chunk 的重排名次虽在前 20，但分数低到几乎不可能被 LLM 用上（中文 0.18 阈值下，它们连"可答"的门槛都过不了）。把它们塞进上下文没有信息价值，这是 K=4 的另一条独立证据。
3. **重排不是简单丢答案，而是"重新洗牌"**：英文粗排 top-4 命中 27 道，重排踢出 4 道、又从 5-20 名捞回 9 道，净增 5 → 32；中文粗排 top-4 原有 11 道，踢出 3 道、捞回 17 道 → 25。Top-1 命中中文从 3 涨到 16，是 MRR 提升的直接来源。
4. **"没进 top-4"现在可以拆开了**（新增三个诊断项，`eval_report.json` 的 `diagnostics` 可直接读）：

| 损失类别 | 含义 | EN | ZH |
| :--- | :--- | :--- | :--- |
| 粗排未召回 | gold 不在粗排 20 里（检索层天花板，扩 K 无用） | 6/50（12%） | 13/50（26%） |
| 窗口截断 | 粗排 5-20 命中、但没进精排 top-4（扩 K 才有救） | 8/50（16%） | 9/50（18%） |
| 重排丢失 | 粗排 top-4 已命中、却被重排挤出 top-4 | 4/27（14.8%） | 3/11（27.3%） |

`fine_recall_at_4_gold_given`（给定粗排 20 命中，精排 top-4 保住率）为 **EN 0.7273 / ZH 0.6757**。这个值与 2026-11-07 旧报告的 `0.7273 / 0.6757` 逐位一致，说明旧口径算的本来就是这件事；而现行 `fine_recall_at_20_gold_given` 因为对粗排 20 条**全量重排**后判断 gold 是否还在列表里（候选集没变），**恒为 1.0**，不能用来证明"重排没丢答案"——它已降级为参考项，与 top-4 口径并列展示。


### 3. 拒答机制评测
| 指标 | EN | ZH |
|---|---|---|
| 可答题数 | 50 | 50 |
| 不可答题数 | 20 | 20 |
| 拒答准确率 | 0.60 | 0.70 |
| 误拒率 | 0.18 | 0.34 |

*误拒率说明：英文 50 道可答题中误拒 9 道（0.18），中文 50 道中误拒 17 道（0.34）。中文偏高是因为 `bge-small-en-v1.5` 为英文单语模型，中文 Query 对英文原文的打分整体偏低，容易跌破中文阈值 0.18。*

**中文阈值 0.15 vs 0.18（同一批分数，正式对比）**

两份报告（`eval_report_zh015.json` / `eval_report_zh018.json`）来自**同一次遍历**的逐题明细，区别只在阈值：召回类指标与阈值无关，因此两列完全相同；差异只出现在拒答类。

| 指标（zh） | 0.15 | 0.18 | 差 |
|---|---|---|---|
| 粗排 Recall@4 / @5 | 0.22 / 0.24 | 0.22 / 0.24 | 不变 |
| 精排 Recall@4 / @5 | 0.50 / 0.52 | 0.50 / 0.52 | 不变 |
| 精排 MRR@20 | 0.4312 | 0.4312 | 不变 |
| 误拒率（可答题被拒） | 0.32（16/50） | 0.34（17/50） | **+1 题** |
| 拒答准确率（不可答题） | 0.65（13/20） | 0.70（14/20） | **+1 题** |
| 　└ unrelated | 1.00 | 1.00 | 不变 |
| 　└ semi_related | 0.375（3/8） | 0.50（4/8） | **+1 题** |
| 　└ counterfactual | 0.50 | 0.50 | 不变 |

结论是**一换一**：0.18 比 0.15 多拦下 1 道半相关题，代价是多误拒 1 道可答题；unrelated 与 counterfactual 在两档下完全不动，说明阈值作用面很窄。选 0.18 的理由是"漏答（幻觉）的代价高于拒答"，而不是"指标整体更优"——如果产品更怕误拒，0.15 同样成立。这也解释了旧报告里"0.70/0.34 vs 0.65/0.32"的差异：**那是同一批样本的两个阈值，不是两次不同数据的运行**（两边召回、MRR、直方图逐项一致，详情见 `eval_report_v1.json` 的 `_note`）。

**拒答准确率（按 category 细分）**：
| 类型 | EN | ZH |
|---|---|---|
| unrelated | 0.75 | 1.00 |
| semi_related | 0.50 | 0.50 |
| counterfactual | 0.50 | 0.50 |

### 关键发现

1. **精排对跨语言场景价值显著**：中文粗排 Recall@4 仅 0.22，重排后拉到 0.50，翻了一倍多。英文也补齐了从 0.54 到 0.64 的截断损失。
2. **中文阈值 0.18 vs 0.15 是一换一，不是"更优"**：同一批分数下，0.18 多拦 1 道半相关不可答题（semi_related 3/8 → 4/8），同时多误拒 1 道可答题（16/50 → 17/50）；unrelated 1.00、counterfactual 0.50 两档不动。选 0.18 的依据是价值排序（漏答代价 > 拒答代价），不是指标净收益。误拒偏高的根因仍在嵌入模型：`bge-small-en-v1.5` 是英文单语模型，中文 Query 对英文原文的整体打分偏低，阈值只能小幅调，治不了根。
3. **截断窗口（K=4）有两条独立证据**：① 分段表里第 5 名对英文的增量是 0、对中文只有 1 题；② 分数分布表里 6-10 段之后的 gold 分数已塌到 0.18 以下（英文 11-20 段中位 0.003），这些"命中"在生成阶段基本不可用。两点合起来，K=5 只买到 1 道中文题，却要多付 25% 的上下文 token。
4. **重排实现了"弱项拯救"与"排序提纯"，但受限于固定窗口**：
   - **在召回上（Top-4）**：重排把英文前 4 名包含的答案数从 27 道提升到了 32 道；中文更是从 11 道直接翻倍到了 25 道（跨语言抢救能力极强）。
   - **在排序上（Top-1）**：英文 Top-1 命中从 11 提升到 15，中文从 3 飙升到 16，这直接导致了 MRR 指标的大幅上涨（中英文均提升 0.08 以上），证明重排把最正确的答案推到了最前面。
   - **天花板已可拆解（口径已补齐）**：`Recall@4 = 0.64 / 0.50` 的缺口现在拆成三块——粗排未召回（EN 6 / ZH 13）、窗口截断（EN 8 / ZH 9）、重排丢失（EN 4/27 / ZH 3/11）。给定粗排 20 命中，精排 top-4 保住率是 EN 0.7273 / ZH 0.6757；而"全量重排后 gold 仍在列表里"那个恒 1.0 的指标已降级为参考项。未来若把粗排 `top_k` 扩到 30/40，只有"窗口截断"那一块能收回来，重排耗时会成倍增加，且中文的截断块（9 题）与粗排未召回块（13 题）比并不划算。

### 已知限制

- **中文粗排弱**：`bge-small-en-v1.5` 为英文单语模型，中文 Query 检索英文 Doc 的 Recall@5 仅 0.24。未来可换用 `BAAI/bge-m3` 等多语言嵌入模型改善。
- **中文误拒率高**：0.34，见上文根因分析，阈值调优无法根治。
- **评估集规模**：每种语言 50 道可答 + 20 道不可答，是「小规模可信、可复现」的规模。指标波动范围约 ±0.04（50 题里错 2 题），建议作为趋势参考而非绝对值。
- **单文档**：当前评估仅基于 `sample.pdf`（LoRA 论文）。多文档、多领域评估是 P2 扩展方向。

---

## 边界分类（12 道争议题）

主评估集的**英文 50 道可答题**中，有 12 道题粗排命中（前 20）但精排未进前 4；按同一判据中文另有 12 道，合计 24 道，本表只列英文。逐条核验后按「单一主分类 + 备注次要影响」归类如下（排名数字已随 `eval_details.json` 落盘，可按 `id` 逐题核对）：

| 主分类 | 题数 | 题号 | 粗排排名 | 精排排名 | 根因 | 备注 |
|---|---|---|---|---|---|---|
| 符号型 | 1 | en_003 | 13 | 16 | `A ∈ R^{d×r}` 数学符号编码弱 | 无 |
| 否定句 | 1 | en_006 | 8 | 11 | 附录 B 反例页挤掉正面结论页 | 无 |
| 表格行标签 | 2 | en_013, en_045 | 14, 13 | 16, 14 | 表格行列关系在 PDF 解析后丢失 | en_044 也含表格，但主分类归「跨页误匹配」 |
| 章节主旨 | 1 | en_040 | 10 | 19 | 元信息类问题，20 个候选分数挤在 0.56–0.61 | 无 |
| 跨页误匹配 | 5 | en_014, en_016, en_029, en_042, en_044 | 12, 2, 2, 4, 3 | 9, 7, 8, 7, 9 | chunk 双页粒度导致机械匹配 | en_044 也含表格因素；en_016 是唯一命中的跨页题 |
| 低信号发布句 | 1 | en_048 | 10 | 16 | 措辞平淡，20 个候选分数 0.48–0.56 | 也可归入跨页误匹配 |
| 关系型语义模糊 | 1 | en_038 | 10 | 7 | 结论在 p2，讨论在 p20–24，两种口径都成立 | 无 |

**这些题是方法能力边界的证据，不是标注错误。** 保留在原评估集里，用于说明「纯向量检索 + Cross-Encoder 精排」在哪些题型上会失效。

---

## 对照组评估（方法擅长领域）

### 对照组的定位

主评估集（140 题）包含 12 道「边界案例」，这些题在方法上天然不擅长。为了验证「方法在擅长领域确实强」，新增对照组 20 道题（中英文各一版，共 40 条记录），中英文语义严格对应。

### 对照组的选题原则

- **类型**：T1 具体数值型、T2 定义型、T3 直接引用型、T4 步骤/配置型、T5 单页内对比型，每类 4 题。
- **避免**：符号型、否定句、表格行标签、章节主旨、跨页误匹配、低信号发布句——这 6 类是主评估集已发现的边界，不在对照组里重复。
- **防自证**：题目由项目所有者出（不看答案位置），`gold_page` 和 `gold_keywords` 由开发伙伴标注，分离「出题」和「找答案」。
- **已知轻微重叠**：`ctrl_en_002` / `ctrl_en_010` / `ctrl_en_016` / `ctrl_en_020`（中文对应 `ctrl_zh_*`）共享 `gold_page=5`（4.2 Practical Benefits 段落），分别测「checkpoint 大小变化」「权重矩阵范围」「显存占用原因」「多任务存储对比」。四道题同页是预期设计，该段是论文的强信号区。

### 对照组的指标

| 指标 | EN | ZH |
|---|---|---|
| 可答题数 | 20 | 20 |
| 粗排 Recall@5 | 0.45 | 0.25 |
| 粗排 MRR@20 | 0.35 | 0.22 |
| 精排 Recall@4 | 0.65 | 0.45 |
| 精排 MRR@4 | 0.45 | 0.43 |

**按 type 分组（EN，每类 4 题）**：

| type | 定义 | 粗排 Recall@5 | 粗排 MRR@20 | 精排 Recall@4 | 精排 MRR@4 | 判断 |
|---|---|---|---|---|---|---|
| T1 | 具体数值型 | 0.75 | 0.64 | **1.00** | 1.00 | ✅ 方法确实擅长 |
| T2 | 定义型 | 0.25 | 0.14 | 0.75 | 0.38 | ⚠️ 粗排弱，精排救回 |
| T3 | 直接引用型 | 0.25 | 0.33 | 0.75 | 0.38 | ⚠️ 同上 |
| T4 | 步骤/配置型 | 0.75 | 0.44 | 0.50 | 0.38 | ⚠️ 粗排强，精排推走 |
| T5 | 单页内对比型 | **0.25** | 0.19 | **0.25** | 0.13 | ❌ 方法完全不擅长 |

### 关键发现

**发现 1：对照组整体 0.45 低于主评估集 0.60，「事实型题方法都擅长」的假设被部分证伪。**

但分组看，T1 精排 Recall@4 = 1.00，方法对「找一个数字」确实满分。而 T5 双指标都是 0.25，方法对「比较两个概念」完全不擅长。**「事实型」不是同质的**：

- T1（找数字）：方法满分。
- T2/T3（找定义/名词）：粗排弱但精排能救回。
- T4（找配置）：粗排强但精排会推走 2 道。
- T5（比差异）：粗排、精排都弱。

**发现 2：T4 的「精排推走」现象与主评估集一致。**

T4 粗排 Recall@5 = 0.75，精排 Recall@4 = 0.50：4 道 T4 题里，粗排命中 3 道、精排保住 2 道，丢 1 道。这是对照组内的具体事实；主评估集层面的「精排丢失率」基于 v1 报告（2026-11-07）的旧口径，已不再引用。

**发现 3：对照组整体低分还受 3 道题拖累（`ctrl_en_004` / `ctrl_en_010` / `ctrl_en_018`，中文各一版）。**

作者判断这 3 道题依赖表格/列表信息，而该题型在主评估集里已知是「表格行标签」边界；出题时没有完全避开，导致对照组整体低分被进一步拉低。**此判断未逐题复核表格来源，标为待核实。**

### 结论

对照组的三层结论：

1. **方法在「找一个数字」上确实满分**（T1 Recall@4 = 1.00），这是方法的核心优势。
2. **方法在「比较概念」「定义型粗排」「配置型精排」上有明确弱点**，这是未来优化方向（混合检索、chunk 切分细化）。
3. **对照组整体 0.45 的原因**：不是方法对所有事实型题都弱，而是「事实型」内部有强弱分化——数值型满分，对比型只有 0.25，五类均值恰好就是整体值 0.45。

这个结论比「假设对照组 0.85 从而验证假设」更有价值——**它说明我们对方法能力边界的认知本身需要数据校准**（0.85 为假设值，非实测）。

---

## 阶段 1–2 的探索与负结果

### 阶段 1：重排消融（`evaluate_ablation.py`）

同一 100 道可答题、同一 top-20 候选池，A 组走重排取 top-4，B 组直接取前 4：

| 指标 | A 带重排（EN / ZH） | B 不重排（EN / ZH） | Δ（EN / ZH） |
|---|---|---|---|
| 精排 Recall@4 | 0.64 / 0.50 | 0.54 / 0.22 | **+0.10 / +0.28** |
| 精排 MRR@4 | 0.43 / 0.40 | 0.35 / 0.12 | **+0.08 / +0.28** |

**关键发现**：重排对中文的贡献是英文的 2.8 倍——BGE-small 是英文单语模型，中文粗排本身就弱（B 组 0.22），重排的多语言能力把它救回来。英文有 9 题被救回、4 题被推走（推走的 4 题全是跨页 chunk 机械匹配，与「边界分类」一致）。

### 阶段 2：混合检索（BM25 + 向量 + RRF），实测无增量，已回滚

设计：英文走 BM25 + 向量两路 RRF 融合，中文只走向量。中文不进 BM25 是设计选择，不是匹配失败：`hybrid_search.py:84-86` 的 `_is_zh()` 负责判定，命中中文即在 `hybrid_search.py:260-261` 短路过 BM25、直接走纯向量；英文 tokenizer 只取 `[a-z0-9]`，中文会被切碎（`hybrid_search.py:28` 亦写明「不引入 jieba」）。

实测（`hybrid_eval_report.json`）：

| 指标（EN） | A 纯向量 | B 混合 | Δ |
|---|---|---|---|
| 粗排 Recall@5 | 0.60 | 0.58 | -0.02 |
| 精排 Recall@4 | 0.64 | 0.64 | 0 |

诊断（`diag_hybrid_report.json`）四条证据：

1. **融合生效但两路同质**：100 题里 50 题 BM25 top-20 与向量 top-20 完全重合。
2. **BM25 排序能力弱**：单独 R@20 = 0.46，仅 1.93 倍随机基线（0.238）。
3. **擅长题型稀缺**：概括型占 34%，数值型受 PDF 提取拖累（`rq=rv=8` 在 84 个 chunk 里 token 数为 0，见 DESIGN.md C34）。
4. **差异是噪声**：McNemar 不一致对仅 4 个（2 救 2 推），样本不足；R@5 差值 95% CI `[-0.08, 0.06]` 含 0（表内 -0.02 是两次点估计之差，诊断报告的 bootstrap 均值是 -0.01）。

**结论**：BM25 的适用前提是「大语料、多文档、关键词精确匹配」，单篇论文（84 chunk）下 IDF 失效、两路同质、擅长题型稀缺。**主动回滚接入**（`qa.py` 改回纯向量），但保留 `hybrid_search.py` 作为多文档场景的现成储备。

**这是一次有数据支撑的负结果，不是「效果不好所以放弃」**：先诊断、量化、再做统计检验，确认是噪声后回滚，并记录根因。

---

## 阶段 3–4：流式与降级

### 阶段 3：流式输出

- **`answer_stream()`**：`qa.py` 新增流式入口，阶段 3 时 yield `token` / `citations` / `refused` 三种事件；阶段 4 增加 `degraded` 后共四种（`qa.py:611-624`）。`answer()` 改为消费它，签名与 `RagAnswer` 结构不变。
- **引用不流式**：引用依赖完整答案里的 `[n]` 标记，流结束后一次性给。
- **首 token 前可重试，已出 token 后不重试**：流式输出不可撤回，中途重试会重复输出；限流通常发生在请求发起时，首 token 前重试覆盖大部分场景。
- **占位提示**：点发送立即 yield "正在检索并重排…"，缓解重排阶段的空白等待（纯 CPU 推理，单请求十几秒量级，随机器性能浮动）。
- **观察**：流式只优化了 LLM 阶段（生成耗时随平台排队波动，从秒级到数十秒），首字延迟仍由重排主导（CPU 上十几秒量级），瓶颈在重排。这个教训记为 Q38。

### 阶段 4：降级与容错

系统依赖三个外部组件：Chroma（向量库）、Cross-Encoder（重排）、智谱 API（生成）。任一失效时不让整页报错，而是降级或给出可读提示。

**五个降级点**：

| 代号 | 触发 | 处理 | 用户可见 |
|---|---|---|---|
| D1 | 重排失败（模型缺失/推理异常） | `degraded` 事件 → 用粗排 top4 继续生成 | ⚠️ 重排暂时不可用 + 答案 |
| D2 | LLM 首 token 前限流，重试一次仍失败 | app.py 捕获 → 友好文案 | "生成服务繁忙，请稍后重试" |
| D3 | 流中途 LLM 中断 | 保留半截答案 + 标注 | 半截答案 + "（生成中断：原因）" |
| D4 | 检索为空 / 阈值拒答 | `refused` 事件 | REFUSAL_TEXT |
| D5 | 重排返回空（防御分支） | `refused(empty)` | REFUSAL_TEXT |

**两条设计取舍**：

1. **D1 降级路径不判阈值**（C3）：粗排余弦分（-1~1）与重排 Cross-Encoder 分（0~1）量纲不同，用同一阈值比会误判。代价是粗排 top4 可能不相关、LLM 仍会生成。
2. **非流式调用方对降级透明**：`answer()` 消费 stream 时遇 `degraded` 静默忽略，调用方无法感知"本次质量下降"。这是"不改 `RagAnswer` 结构"的必然代价，阶段 6 日志补可观测性。

**已知限制**：

- **免费档 GLM 高峰期慢**：本机 `logs/app.log`（2026-11-18）7 条完整请求的总耗时从十几秒到三十多秒，其中 LLM 占大头（秒级到数十秒）。旧文曾记一次 92 秒总耗时及其分项，但分项相加不成立、也无法在日志中复现，已不再引用。这是平台侧，不是代码问题；流式下用户能看到逐字出，但首 token 仍可能等很久。
- **占位提示不显示用时/阶段**：点发送后显示"正在检索并重排…"，但用户不知道卡在哪一步、还要等多久。分阶段文案（检索→重排→生成→重试）和计时记为后续优化项（S8）。

---

## CI/CD 与未来演进

本项目的自动化流水线设计分为**当前（单人）**和**未来（多人）**两个阶段。

### 两条流水线，环境不同

本项目上线涉及**两条不同的流水线**，环境并不等价，复现时需分开看：

- **GitHub Actions `ci.yml`（测试）**：`ubuntu-latest` + Python 3.13，`pip install -r requirements.txt` 后再装 `pytest`/`modelscope`；依赖中的 `torch` 取 PyPI 默认（Linux 上为 CUDA 版，体积远大于 CPU 版），与运行时镜像不同。模型靠 `actions/cache` 缓存 `./models`，缓存键固定为 `models-${{ runner.os }}-v1`。
- **魔搭侧 Docker 构建（部署）**：`Dockerfile` 先从 PyTorch CPU 源单独装 `torch==2.15.0`，再按 `requirements.txt` 装其余依赖，因此运行时是纯 CPU 环境。

**云端构建实测（2026-11-14，云效 → 魔搭 ACR）**：`python:3.13-slim` 基础镜像 + `torch 2.15.0+cpu`（209.5 MB wheel）+ `requirements.txt` 全量安装成功（含 `pdfplumber-0.11.8` 与 `Pillow-11.3.0`），镜像推送完成；整个构建分钟级、全程 CPU。

**2026-11-22 复测（GitHub Actions 推送到魔搭仓库 → 云效构建 → 自动部署）**：`Dockerfile` 的 pip 已改为「清华源优先、不可达时回退官方 PyPI」。构建日志里 `#10` 用 75.7 秒装完 `requirements.txt`（`Successfully installed ... Pillow-11.3.0 ...`），随后 `exporting layers` → `loading layer` → 镜像推送成功；同日自动部署链路验证通过（CI 轮询到 `Running`，且 `DeployedTime` 晚于本次 push 时间）。

### 单人阶段的设计

- **触发条件**：`push` 到 `master` 分支。
- **测试范围**：`test_ingest.py` + `test_index_store.py` + 纯逻辑测试。**不跑**需要真实 `ZHIPU_API_KEY` 的 API 测试。
- **部署链路**：测试通过（`test` Job）→ 原生 `git fetch` 拉取并覆盖 → 打破 Docker 缓存 → `git push` 同步到魔搭创空间 → 触发平台自动构建 → **轮询空间状态直到上线**：用公开只读接口读 `Status` 与 `DeployedTime`，要求 `DeployedTime` 晚于本次 push 时间才判成功；出现 `*Failed*` 或 30 分钟内未上线则 CI 变红。
- **设计考量**：
  1. 不跑真实 API 测试，避免免费档 429 随机变红导致无法部署。
  2. 用原生 Git 命令结合文件覆盖，绕开魔搭的保护分支（Protected Branch）和 LFS 残留限制。
  3. 修改 `requirements.txt` 强制打破镜像层缓存，确保每次构建使用的是最新代码。
  4. 部署不再「推完就走」：`deploy` Job 末尾轮询空间状态，把「构建或启动失败」变成 CI 红灯。判据要求 `DeployedTime` 晚于本次 push 时间，避免旧实例仍在 `Running` 时被误判成新版本已上线。
  5. 同一分支加 `concurrency` 组且 `cancel-in-progress: true`：新 push 自动取消仍在跑的旧 run。否则两个 run 会同时向魔搭推送，构建排队叠成两个，耗时翻倍（实测过一次）。

### 演进原则

1. **不重写骨架，只加门禁**：从 `push` 触发升级到 `pull_request` 触发，测试 Job 保持不变，只增加部署的审批条件。
2. **测试与部署分离**：测试不过不部署是底线。
3. **Secrets 最小权限**：单人用 Repository Secrets；多人必须迁移到 Environment Secrets。

### 演进方法（从单人切多人）

**触发层**：将 `on: push` 扩展为 `on: pull_request`。开启 Branch Protection，要求合并前必须通过 Status Checks 和至少 1 个 Approve。

**部署层**：在 GitHub Settings 配置 **Environments**（如 `production`），设置 `Required reviewers`。在 `ci.yml` 的 `deploy` Job 绑定 `environment: production`。这样即使 CI 全绿，也会暂停等人点“Approve”才执行部署。

**权限层**：将 `MODELSCOPE_TOKEN` 从 Repository Secrets 迁移到 Environment Secrets。添加 `.github/CODEOWNERS` 文件，指定核心目录必须由 Owner 审阅。

### 演进代价

- **审批增加延迟**：每次合并主干需等人点 Approve，紧急修复 bug 会被卡住。
- **分支保护变严**：不能再随手 `git push --force` 覆盖主干，必须走 PR。
- **Secrets 迁移有成本**：迁移后本地旧配置可能失效，需要重新配置。

---

## 未来工作

- **nDCG@K**：当前用 Recall + MRR，未引入需要人工标注相关性等级的 nDCG。如果未来评估集扩大到 200 题以上，可引入 nDCG@5 衡量排序质量。
- **RAGAS 框架**：当前手写评估脚本，只覆盖检索侧指标。生成侧 Faithfulness 评估可用 RAGAS（需要 LLM 调用，成本较高）。
- **多语言嵌入模型**：换 `BAAI/bge-m3` 后重新评估，预期中文粗排 Recall@5 从 0.24 显著提升。
- **混合检索（多文档场景）**：单文档实测无增量已回滚（见「阶段 1–2 的探索与负结果」）。多文档/大语料下 IDF 恢复区分度，`hybrid_search.py` 可直接复用，届时重跑 `evaluate_hybrid.py` 验证。
- **chunk 切分细化**：当前 400 token 一段、相邻块重叠 80 token（滑窗步长 320，`ingest.py:134`），可尝试 256 或 320 token 一段，减少跨页 chunk 的粒度，改善「跨页误匹配」。
- **对比型问题的针对性处理**：T5 双指标 0.25，说明「比较两个概念」需要专门的检索策略（比如拆成两个子问题分别检索，再合并）。
- **重排耗时优化（S7）**：单请求重排在 CPU 上为十几秒量级（本机 7 条完整请求，2026-11-18），另有一次冷启动明显更慢（未进入 LLM）；是首字延迟主要来源。候选：换更小 reranker、减 top_k、量化、CPU 线程调优。
- **分阶段文案 + 计时（S8）**：占位提示升级为"正在检索…→正在重排…→正在生成…"，并显示已等待秒数。

---

## License

本项目以 Apache License 2.0 授权（见文件顶部 front matter）。

注意：**仓库根目录当前没有 `LICENSE` 文件**（`00_tree.txt` 与实测均无），发布前需补齐，否则该声明缺少实际授权文本。
