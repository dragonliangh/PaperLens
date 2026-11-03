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

> 当前进度：**第 1–5 块已完成并通过验收**（解析分块 → 向量索引 → 检索生成引用 → Gradio 界面 → Docker 部署）。

---

## 目录结构

```text
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
├── app.py                # 第 4 块：Gradio 界面
├── Dockerfile            # 第 5 块：Docker 构建文件
├── docker.yaml           # 魔搭创空间配置
├── ms_deploy.json        # 魔搭 Docker 部署配置
├── requirements.txt      # 依赖包版本
├── README.md             # 本文件
└── DESIGN.md             # 契约、设计决策、变更记录

环境要求
Windows 10 / 11（本项目在 Win10 + Python 3.13 上开发）

Python 3.13

依赖：

第 1 块：pdfplumber、transformers、torch（CPU 版）、pytest

第 2 块：chromadb、sentence-transformers

第 3 块：zhipuai、sniffio

第 4 块：gradio

首次配置
创建并激活虚拟环境（VS Code 通常会自动创建 .venv）：

powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& d:\python\PaperLens\.venv\Scripts\Activate.ps1
配置 pip 国内镜像（只需一次）：

powershell
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
安装依赖：

powershell
pip install pdfplumber transformers torch pytest chromadb sentence-transformers zhipuai sniffio gradio -i https://pypi.tuna.tsinghua.edu.cn/simple
设置智谱 API Key（第 3 块需要）：

powershell
$env:ZHIPU_API_KEY = "你的key"
离线模型
本项目不联网加载模型。模型文件已提前下载到 models/bge-small-en-v1.5/，ingest.py 和 index_store.py 直接读本地路径。

如果换一台机器、这个目录不存在，可使用 ModelScope 下载：

python
from modelscope import snapshot_download
snapshot_download('BAAI/bge-small-en-v1.5', local_dir='./models/bge-small-en-v1.5')
运行
本地运行测试
powershell
pytest test_ingest.py test_index_store.py test_qa.py -v
预期：15 passed（需已设置 $env:ZHIPU_API_KEY）。

本地启动 Web 界面
powershell
python app.py
浏览器打开控制台输出的本地地址（通常是 http://127.0.0.1:7860）。

已知限制
这些是有意接受的边界，不是 bug，详见 DESIGN.md。

第 1 块：PDF 解析
PDF 双栏排版可能导致文字顺序错乱；公式、表格抽取不保证正确。

单词间空格可能丢失（如 Edward Hu → EdwardHu）。

章节识别不保证成功，section 可能为空。

第 2 块：向量索引
build_index 假设一批 chunks 属于同一篇论文。

score = 1 - cosine_distance 是相对排序依据，不是概率。

纯向量检索对「语义相关但未直接回答」的段落区分度有限，这是 P1 引入重排的直接动机。

第 3 块：LLM 生成
LLM 引用可能错位：机制本身正常，是 LLM 标注不精确。这是 RAG 已知问题，由第 7 块 Faithfulness 评估量化。

免费档 completion_tokens 可能高于可见答案长度，属平台计费口径，不影响功能。

第 4 块：Gradio 界面
上传的 PDF 存在 Gradio 的临时目录，进程重启即失效。

索引是全局的：多个用户上传不同论文会共存于同一 Collection，检索时跨论文混合。

P0 不做流式输出，期间界面显示处理中。

第 5 块：Docker 部署
魔搭创空间磁盘非持久化，应用重启时会自动重新索引 data/sample.pdf。

免费资源在高峰期生成中文长句时，偶尔可能超时。

路线图
块	内容	状态
1	PDF 解析 + 分块	✅ 完成
2	嵌入 + 向量库索引（Chroma）	✅ 完成
3	检索 + 生成 + 引用（GLM-4.7-Flash）	✅ 完成
4	Gradio 界面	✅ 完成
5	Docker 化 + 魔搭创空间部署	✅ 完成
6	两阶段检索（重排，P1）	⬜ 待开始
7	评估脚本 + 拒答机制（P1）	⬜
8	日志 / CI / 架构图 / 消融（P2）	⏸ 以后再说
