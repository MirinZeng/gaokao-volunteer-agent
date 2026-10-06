<div align="center">

# 🎓 高考志愿填报 RAG Agent

**基于真实招生 Excel、结构化检索、混合 RAG 与四层会话记忆的智能志愿决策助手**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-Agent-1C3C3C)](https://python.langchain.com/)
[![Vue](https://img.shields.io/badge/Vue-3-42B883?logo=vuedotjs&logoColor=white)](https://vuejs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Streaming-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![DeepSeek](https://img.shields.io/badge/LLM-DeepSeek-4D6BFE)](https://www.deepseek.com/)
[![License](https://img.shields.io/badge/Visibility-Private-lightgrey)](#)

从分数/位次换算，到冲稳保垫分层、专业组调剂风险和联网核验，提供可解释、可追溯的志愿建议。

</div>

---

这是一个带 Vue 对话界面的 Python/LangChain 高考志愿填报 Agent。它以项目中的 Excel
作为招生数据事实底座，并使用：

- SQLite 结构化招生表（分数、位次、科类、院校、专业、城市等精确筛选）
- SQLite FTS5 BM25（核心字段中文分词、关键词召回）
- 智谱 `embedding-3` + FAISS 内积索引（向量先归一化，因此等价于余弦相似度）
- 加权混合检索
- 按需联网核验（Tavily 可选，未配置时回退 DuckDuckGo；优先官方来源）
- DeepSeek OpenAI 兼容接口生成答案
- Excel 原始行号引用与“证据不足则拒答”提示词约束
- 输出引用白名单校验：模型引用不存在或未检索到的 Excel 行时，直接阻止展示

## 1. 填写 API Key

项目根目录已创建 `.env` 占位配置；也可以从模板重新复制：

```powershell
Copy-Item .env.example .env
```

然后编辑 `.env`：

```dotenv
DEEPSEEK_API_KEY=在这里填写你的DeepSeek_API_Key
ZHIPU_API_KEY=在这里填写你的智谱_API_Key
```

`.env` 已被 `.gitignore` 排除，不会被提交。默认 DeepSeek 模型为当前接口中的 `deepseek-flash`；也可以通过 `DEEPSEEK_MODEL` 改成账户可用的其他模型。

## 2. 建立索引

首次使用或 Excel 更新后执行：

```powershell
.\.venv\Scripts\python.exe -m gaokao_agent ingest
```

如果暂时不使用智谱 Embedding，可以只建立 BM25 索引：

```powershell
.\.venv\Scripts\python.exe -m gaokao_agent ingest --bm25-only
```

纯 BM25 模式不会调用智谱 API。之后的 `ask`、`chat` 和 `main.py` 会自动识别该模式。

已有索引只想重建结构化数据库和精简 BM25、不重算 Embedding：

```powershell
.\.venv\Scripts\python.exe -m gaokao_agent rebuild-search
```

每个 Excel 数据行仍是一条完整证据；BM25/Embedding 只使用院校、专业、科类、
城市、分数和位次等核心字段，避免长篇详情被硬切后平均而稀释检索信号。

联网只用于当年政策、招生章程、计划变化、调剂和退档规则等时效信息；院校、
专业、历史分数与位次仍以本地 Excel/SQLite 为事实底座。前端可按轮关闭联网核验。

对“某位次推荐某专业”这类问题，程序先用 SQLite 做科类与位次硬筛选，再按最近一年
最低位次与考生位次的比例分为“冲 / 稳 / 保 / 垫”。命中一个目标专业后，还会按
“科类 + 院校 + 专业组代码”展开同组其他专业，供模型判断服从调剂时是否可能进入
土木、园林等非目标方向。回答默认给出 10~16 个代表方案、逐项选择理由与主要风险，
并在信息不足时追问就业、城市、学校层次、预算及调剂接受度等偏好。

这一步会调用智谱 Embedding API，并在 `data/index/` 生成本地 FAISS 索引、文档数据和清单。约 3.9 万条记录会产生 API 用量，执行前请确认账户余额。若中途失败，程序会保留内存映射的增量 embedding 检查点；重新运行即可继续，不会每批重写全部历史向量。

程序会同时按条数和总字符数动态拆分 Embedding 请求；如果智谱仍返回输入过长的
`400/1210`，还会自动二分失败批次。调整 `.env` 中的
`EMBEDDING_MAX_BATCH_CHARS` 可以进一步降低单次请求大小。

## 3. 提问

单次提问：

```powershell
.\.venv\Scripts\python.exe -m gaokao_agent ask "江西物理类，位次 30000 左右，有哪些计算机相关专业可参考？"
```

回答采用 JSON Lines 流式协议：`start` 表示检索完成，`delta` 是模型增量，
`result` 是经过 Excel 引用白名单校验的最终结构化结果。例如：

```json
{"type":"start","status":"retrieved","retrieval_mode":"bm25","evidence_count":8}
{"type":"delta","content":"根据表内数据……","validated":false}
{"type":"result","status":"ok","answer":"……","citations":[35087],"evidence":[{"excel_row":35087}],"retrieval_mode":"bm25","validated":true}
```

流式 `delta` 尚未完成引用校验，因此固定标记为 `validated:false`；业务程序应以最后一条
`result` 为准。

默认 `DEEPSEEK_TEMPERATURE=0.4`。Excel 是具体招生数据的优先依据；模型也可以补充明确
标记为“一般性建议（非Excel数据）”的通用分析。最终 JSON 的 `grounding` 字段会区分
`excel_supported` 与 `general_guidance`。

连续对话：

```powershell
.\.venv\Scripts\python.exe -m gaokao_agent chat
```

在 PyCharm、VS Code 等 Python 工作台中，也可以把项目解释器设置为
`.venv\Scripts\python.exe`，然后直接运行根目录的 `main.py`。不传参数时，
`main.py` 会自动进入连续对话模式。

只测试检索证据、不调用 DeepSeek：

```powershell
.\.venv\Scripts\python.exe -m gaokao_agent search "南昌大学 软件工程 物理"
```

检查 Excel：

```powershell
.\.venv\Scripts\python.exe -m gaokao_agent inspect-excel
```

## 数据边界

Excel 是院校、专业、分数、位次、学费和选科等具体招生数据的首要依据。每条证据带有 `[Excel第N行]` 标识。模型也可以补充明确标记为“一般性建议（非Excel数据）”的通用分析，但不得把它伪装成表内数据或当年官方政策。

这份数据是志愿研究材料，不代表录取承诺。分数、位次、招生计划和政策可能变化，最终填报前仍需以考试院与院校当年官方信息核验。

## 项目结构

```text
gaokao_agent/
  agent.py          # LangChain RAG 回答链
  cli.py            # 命令行入口
  config.py         # .env 配置
  embeddings.py     # 智谱 Embeddings 适配器
  excel_loader.py   # Excel 读取与证据行构造
  indexer.py        # 可断点续建的 FAISS 索引
  retriever.py      # SQLite 结构化过滤 + BM25 + FAISS 余弦相似度混合检索器
tests/              # 不需要 API Key 的单元测试
frontend/           # Vue 3 + Vite 聊天界面
```

## Vue 前端

会话记忆使用 `data/memory.sqlite3` 本地持久化，并按会话 ID 严格隔离为四层：

1. 会话元数据：标题、创建/更新时间、消息数量；
2. 用户结构化档案卡：省份、科类、分数/位次、专业与城市偏好、预算、调剂等；
3. 近期对话摘要：把滑动窗口之外的旧消息压缩保存，避免上下文无限增长；
4. 滑动窗口：保留最近 12 条原始消息，供意图理解和代词追问使用。

前端刷新后会从后端恢复会话列表、当前会话、历史消息和档案；不同会话不再共享
同一份 Agent `history`。删除侧边栏会话时，其档案、摘要和消息会在 SQLite 中级联删除。

先启动 FastAPI 后端：

```powershell
cd D:\Gaokaozhiyuan_Agent
.\.venv\Scripts\python.exe -m uvicorn gaokao_agent.api:app --reload --port 8000
```

再开一个 PowerShell 窗口启动 Vue：

```powershell
cd D:\Gaokaozhiyuan_Agent\frontend
npm install
npm run dev
```

浏览器打开 `http://127.0.0.1:5173`。开发服务器会把 `/api` 自动代理到 FastAPI 的 8000 端口。

日常使用可以先构建一次前端，然后只启动一个服务：

```powershell
cd D:\Gaokaozhiyuan_Agent\frontend
npm run build
cd ..
python main.py web
```

浏览器打开 `http://127.0.0.1:8000`。

