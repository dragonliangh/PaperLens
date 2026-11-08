---
domain: 
tags: 
datasets: 
  evaluation:
  test:
  train:
models: 
license: Apache License 2.0
---

# PaperLens —— 学术论文 RAG 助手

上传学术论文 PDF，基于原文回答技术问题，并给出**页码级引用**；检索不到证据时**明确拒答**，而不是强行生成。

> 当前进度：**第 1–7 块已完成并通过验收**（解析分块 → 向量索引 → 检索生成引用 → Gradio 界面 → Docker 部署 → 两阶段检索 → 评估与拒答 → 对照组评估）。

---

## 目录结构

```text
PaperLens/
├── data/
│   ├── sample.pdf              # 样例论文（LoRA），用于测试与演示
│   └── chroma/                 # Chroma 持久化目录（第 2 块自动生成）
├── models/
│   ├── bge-small-en-v1.5/      # 第 1/2 块：嵌入模型
│   └── bge-reranker-base/      # 第 6 块：重排模型
├── ingest.py                   # 第 1 块：PDF 解析 + 分块
├── test_ingest.py              # 第 1 块验收测试
├── index_store.py              # 第 2 块：嵌入 + 向量库索引
├── test_index_store.py         # 第 2 块验收测试
├── qa.py                       # 第 3/6/7 块：检索 + 重排 + 生成 + 引用 + 拒答
├── test_qa.py                  # 第 3 块验收测试
├── rerank.py                   # 第 6 块：Cross-Encoder 精排
├── app.py                      # 第 4 块：Gradio 界面
├── evaluate.py                 # 第 7 块：主评估脚本（140 题）
├── evaluate_ctrl.py            # 第 7 块补充：对照组评估（20 题 × 2 语言）
├── test_eval.py                # 第 7 块验收测试
├── eval_set.json               # 第 7 块：140 条主评估集
├── ctrl_eval_set.json          # 第 7 块补充：40 条对照组评估集
├── thresholds.json             # 第 7 块：按语言的拒答阈值
├── eval_report.json            # 第 7 块：主评估输出（自动生成）
├── ctrl_eval_report.json       # 第 7 块补充：对照组评估输出（自动生成）
├── test_final.py               # 本地模型修复脚本（从 .bin 重生成 .safetensors）
├── Dockerfile                  # 第 5 块：Docker 构建文件
├── docker.yaml                 # 魔搭创空间配置
├── ms_deploy.json              # 魔搭 Docker 部署配置
├── requirements.txt            # 依赖包版本（锁定）
├── README.md                   # 本文件
└── DESIGN.md                   # 契约、设计决策、变更记录
```

---

## 环境要求

- Windows 10 / 11（本项目在 Win10 + Python 3.13 上开发）
- Python 3.13

### 依赖

- 第 1 块：`pdfplumber`、`transformers`、`torch`（CPU 版）、`pytest`
- 第 2 块：`chromadb`、`sentence-transformers`
- 第 3 块：`zhipuai`、`sniffio`（zhipuai 漏声明的依赖，需手动补装）
- 第 4 块：`gradio`（**锁定 5.49.1**，原因见第 5 块已知限制）
- 第 5 块：`pdfplumber==0.11.8`、`Pillow==11.3.0`、`gradio==5.49.1`（解决 Pillow 冲突）
- 第 6 块：复用 `sentence-transformers`，新增模型文件 `models/bge-reranker-base/`
- 第 7 块：无新增依赖（纯标准库）

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
   pip install pdfplumber transformers torch pytest chromadb sentence-transformers zhipuai sniffio gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

4. 设置智谱 API Key（第 3 块需要）：

   ```powershell
   $env:ZHIPU_API_KEY = "你的key"
   ```

### 离线模型

本项目**不联网加载模型**。两个模型文件已提前下载：

- `models/bge-small-en-v1.5/`：第 1 块 tokenizer + 第 2 块嵌入模型。
- `models/bge-reranker-base/`：第 6 块 Cross-Encoder 重排模型。

如果换一台机器、这些目录不存在，可使用 ModelScope 下载：

```python
from modelscope import snapshot_download
snapshot_download('BAAI/bge-small-en-v1.5', local_dir='./models/bge-small-en-v1.5')
snapshot_download('BAAI/bge-reranker-base', local_dir='./models/bge-reranker-base')
```

---

## 运行

### 本地运行测试

```powershell
pytest test_ingest.py test_index_store.py test_qa.py -v
```

预期：`test_ingest` 5 passed、`test_index_store` 7 passed、`test_qa` 5 passed（需已设置 `$env:ZHIPU_API_KEY`）。

```powershell
pytest test_eval.py -v
```

预期：10 passed（需先建索引，约 2 分钟）。

### 本地建索引

```powershell
python -c "from ingest import parse_and_chunk; from index_store import build_index; build_index(parse_and_chunk('data/sample.pdf')); print('ok')"
```

### 本地启动 Web 界面

```powershell
python app.py
```

浏览器打开控制台输出的本地地址（通常是 `http://127.0.0.1:7860`）。

### 跑评估

```powershell
python evaluate.py
```

约 23 分钟（CPU 上 140 题 × 粗排 + 精排）。输出表格 + 写入 `eval_report.json`。

```powershell
python evaluate_ctrl.py
```

约 1-2 分钟（40 条对照组）。输出表格 + 写入 `ctrl_eval_report.json`。

---

## 已知限制

这些是有意接受的边界，不是 bug，详见 `DESIGN.md`。

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

### 第 4 块：Gradio 界面

- 上传的 PDF 存在 Gradio 的临时目录，进程重启即失效。
- 索引是全局的：多个用户上传不同论文会共存于同一 Collection，检索时跨论文混合。
- P0 不做流式输出，期间界面显示处理中。

### 第 5 块：Docker 部署（魔搭）

- **依赖版本冲突**：`pdfplumber 0.11.10` 强制要求 `Pillow>=12.0`，而 `gradio 5.49.1` 要求 `Pillow<12.0`。解决方案：降级 `pdfplumber` 到 `0.11.8`，并显式锁定 `Pillow==11.3.0`。
- **Gradio 版本**：Gradio 6.x 在魔搭的 iframe 环境中存在渲染兼容性问题（聊天框不显示内容），降级到 `5.49.1` 解决，且必须显式声明 `type="messages"`。
- **镜像体积**：Linux 下默认安装的 `torch` 会附带大量 CUDA 依赖（约 3.3GB）。解决方案：单独从 PyTorch CPU 源安装 `torch==2.15.0`，镜像体积缩减至约 1.1GB。
- **性能表现**：实测云端检索耗时约 0.03 秒，GLM 生成耗时 7-10 秒。前端 Gradio 渲染可能会额外增加等待感。

### 第 6 块：两阶段检索与云端修复

- **两阶段检索**：向量粗排（top-20）→ Cross-Encoder 精排（top-4）。实测重排后，含答案的摘要页 p1 从粗排第 4 名升至第 1 名（分数从 0.76 提升到 0.89）。
- **重排分数**：Cross-Encoder 输出的分数经 Sigmoid 归一化到 0~1，**与余弦相似度量纲不同**。第 7 块设拒答阈值时必须使用重排分数。
- **CPU 性能**：`bge-reranker-base` 在 CPU 上必须限制 `batch_size=4`，否则内存交换会让单次重排从 1 秒飙升到 25 秒。
- **云端模型加载**：曾遭遇两个问题——(1) Git LFS 传输导致 `.safetensors` 文件头损坏；(2) 强制走 `.bin` 又触发 PyTorch 2.6+ 的 `weights_only=True` 安全限制。**最终方案**：本地用 `test_final.py` 从 `.bin` 重新生成干净的 `.safetensors` 并推送到云端，让云端回归 `safetensors` 加载路径，完全绕开 `torch.load`。

---

## 路线图

| 块 | 内容 | 状态 |
|---|---|---|
| 1 | PDF 解析 + 分块 | ✅ 完成 |
| 2 | 嵌入 + 向量库索引（Chroma） | ✅ 完成 |
| 3 | 检索 + 生成 + 引用（GLM-4.7-Flash） | ✅ 完成 |
| 4 | Gradio 界面 | ✅ 完成 |
| 5 | Docker 化 + 魔搭创空间部署 | ✅ 完成 |
| 6 | 两阶段检索（重排，P1） | ✅ 完成 |
| 7 | 评估脚本 + 拒答机制（P1） | ✅ 完成 |
| 7+ | 对照组评估（P1 补充） | ✅ 完成 |
| 8 | 日志 / CI / 架构图 / 消融（P2） | ⏸ 以后再说 |

---

## 评估结果

### 评估集设计

- **规模**：140 条（英文 70 + 中文 70，中英文严格语义对应，id 一一对应）。
- **可答题**：每种语言 50 道，`gold_page` 人工标注，命中判定为「`gold_page` 落在 chunk 的 `[page_start, page_end]` 区间内」。
- **不可答题**：每种语言 20 道，按 2:2:1 结构设计——8 道完全无关（unrelated）、8 道半相关（semi_related）、4 道反事实（counterfactual）。
- **中文题处理**：中文题与英文题语义严格对应，`gold_page` 和 `gold_keywords` 与英文完全一致（keywords 用英文原词，因为检索池是英文原文）。

### 指标口径

| 指标 | 口径 |
|---|---|
| 粗排 Recall@5 | 基于 `search()` 的 top-20 结果，只看前 5 个是否命中 `gold_page` |
| 粗排 MRR@20 | 基于 `search()` 的 top-20 结果，第一个命中的排名倒数 |
| 精排 Recall@4 | 基于 `rerank()` 的 top-4 结果，前 4 个是否命中 |
| 精排 MRR@4 | 基于 `rerank()` 的 top-4 结果，第一个命中的排名倒数 |
| 拒答准确率 | 不可答题中，重排最高分低于阈值（判定拒答）的比例 |
| 误拒率 | 可答题中，重排最高分低于阈值（错误拒答）的比例 |

**阈值**：英文 0.3，中文 0.18（跨语言匹配分数天然偏低，阈值单独下调）。

### 评估结果表

| 指标 | EN | ZH |
|---|---|---|
| 可答题数 | 50 | 50 |
| 不可答题数 | 20 | 20 |
| 粗排 Recall@5 | 0.60 | 0.24 |
| 粗排 MRR@20 | 0.39 | 0.18 |
| 精排 Recall@4 | 0.64 | 0.50 |
| 精排 MRR@4 | 0.43 | 0.40 |
| 拒答准确率 | 0.60 | 0.70 |
| 误拒率 | 0.18 | 0.34 |

**拒答准确率（按 category 细分）**：

| 类型 | EN | ZH |
|---|---|---|
| unrelated | 0.75 | 1.00 |
| semi_related | 0.50 | 0.50 |
| counterfactual | 0.50 | 0.50 |

### 关键发现

1. **精排对跨语言场景价值显著**：中文粗排 Recall@5 只有 0.24，但精排 Recall@4 拉到 0.50，翻了一倍。这直接证明两阶段检索的必要性——粗排因嵌入模型单语缺陷漏掉大量答案，精排凭多语言能力把它们捞回来。

2. **中文误拒率偏高（0.34）**：根因是 `bge-small-en-v1.5` 为英文单语模型，中文 Query 与英文 Doc 的匹配分数整体偏低，导致「相关但不回答」的题与「真正相关」的题分数接近，阈值调优无法根治。实测阈值从 0.18 降到 0.15，误拒率只改善 2 个百分点，但拒答准确率损失 5 个百分点，半相关题损失 12.5 个百分点，得不偿失，最终回调到 0.18。

3. **精排从粗排找回的结果中仍有丢失**：英文粗排 Recall@20 = 0.88，但精排 Recall@4 只保住 0.73（丢失 15%）；中文粗排 Recall@20 = 0.74，精排保住 0.68（丢失 6%）。这说明 Cross-Encoder 并非在所有题上都优于向量检索，部分题的精排分数会低于粗排。已在 `eval_report.json` 的 `diagnostics.fine_recall_at_20_gold_given` 中持续监控。

### 已知限制

- **中文粗排弱**：`bge-small-en-v1.5` 为英文单语模型，中文 Query 检索英文 Doc 的 Recall@5 仅 0.24。未来可换用 `BAAI/bge-m3` 等多语言嵌入模型改善。
- **中文误拒率高**：0.34，见上文根因分析，阈值调优无法根治。
- **评估集规模**：每种语言 50 道可答 + 20 道不可答，是「小规模可信、可复现」的规模。指标波动范围约 ±0.04（50 题里错 2 题），建议作为趋势参考而非绝对值。
- **单文档**：当前评估仅基于 `sample.pdf`（LoRA 论文）。多文档、多领域评估是 P2 扩展方向。

---

## 边界分类（12 道争议题）

主评估集 140 题中，有 12 道题在粗排命中但精排丢失，经逐条核验后按「单一主分类 + 备注次要影响」归类如下。

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
- **已知轻微重叠**：`ctrl_002` / `ctrl_016` / `ctrl_020` 共享 `gold_page=5`（4.2 Practical Benefits 段落），分别测「单 checkpoint 大小」「显存占用原因」「多任务存储总量」。三道题同页是预期设计，该段是论文的强信号区。

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

T4 粗排 Recall@5 = 0.75，精排 Recall@4 = 0.50，粗排找到的 3 道题里精排丢了 1 道。这与主评估集诊断的「精排从粗排命中中丢 15%」是同一问题——Cross-Encoder 在部分题上不如向量检索。

**发现 3：对照组含 3 道表格题（ctrl_004 / ctrl_010 / ctrl_018），拖低了整体分数。**

这 3 道题在主评估集里已知是「表格行标签」问题，出题时没有完全避开，导致对照组整体低分被进一步拉低。**这不是对照组设计失败，而是暴露了「方法擅长的事实型题」这个假设本身包含表格型**——表格型恰恰是方法不擅长的。

### 结论

对照组的三层结论：

1. **方法在「找一个数字」上确实满分**（T1 Recall@4 = 1.00），这是方法的核心优势。
2. **方法在「比较概念」「定义型粗排」「配置型精排」上有明确弱点**，这是未来优化方向（混合检索、chunk 切分细化）。
3. **对照组整体 0.45 的原因**：不是方法对所有事实型题都弱，而是「事实型」内部有强弱分化——数值型满分，对比型接近 0，平均下来就是 0.45。

这个结论比「对照组 0.85 验证假设」更有价值——**它说明我们对方法能力边界的认知本身需要数据校准**。

---

## 未来工作

- **nDCG@K**：当前用 Recall + MRR，未引入需要人工标注相关性等级的 nDCG。如果未来评估集扩大到 200 题以上，可引入 nDCG@5 衡量排序质量。
- **RAGAS 框架**：当前手写评估脚本，只覆盖检索侧指标。生成侧 Faithfulness 评估可用 RAGAS（需要 LLM 调用，成本较高）。
- **多语言嵌入模型**：换 `BAAI/bge-m3` 后重新评估，预期中文粗排 Recall@5 从 0.24 显著提升。
- **混合检索**：引入 BM25 + 向量混合检索，可能改善「符号型」「表格行标签」类问题。
- **chunk 切分细化**：当前 400 token 一段，可尝试 256 或 320，减少跨页 chunk 的粒度，改善「跨页误匹配」。
- **对比型问题的针对性处理**：T5 双指标 0.25，说明「比较两个概念」需要专门的检索策略（比如拆成两个子问题分别检索，再合并）。
