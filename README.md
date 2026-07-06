# 1-pocketflow_rag

基于 [PocketFlow](https://github.com/The-Pocket/PocketFlow) 的 RAG（检索增强生成）示例项目。

## 项目结构

```
├── main.py                 # CLI 演示入口
├── requirement.txt         # 依赖列表
├── README.md
└── Server_Ask/
    ├── start_server.py     # FastAPI 服务入口
    ├── middleware.py        # Session 中间件
    ├── common/
    │   ├── utils.py        # 工具函数（LLM 调用、Embedding）
    │   └── defaults.py     # 默认文档配置
    ├── rag_logic/
    │   ├── flow.py         # 离线/在线流程定义
    │   └── nodes.py        # PocketFlow 节点实现
    └── sessions/
        └── session.py      # 会话管理
```

## 功能

- **离线流程**：文档切片 → Embedding 向量化 → FAISS 索引构建
- **在线流程**：查询向量化 → 相似文档检索 → LLM 生成回答
- **会话管理**：多会话隔离，每个 session 独立的文档和索引
- **文档上传**：支持上传自定义文档到指定 session

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirement.txt
```

### 2. 配置环境变量

编辑 `Server_Ask/.env` 文件，填入你的 API Key。

### 3. 运行 CLI 演示

```bash
python main.py --你的问题
```

### 4. 启动 API 服务

```bash
cd Server_Ask
python start_server.py
```

服务启动后访问 `http://127.0.0.1:23333`

## API 接口

| 方法 | 路径 | 说明 | 参数 |
|------|------|------|------|
| GET | `/` | 首页 | 无 |
| POST | `/chat` | 发送问题 | `query: str`, `session_id?: str` |
| POST | `/upload` | 上传文档 | `texts: list[str]`, `session_id?: str` |

### 示例：对话

```bash
curl -X POST http://127.0.0.1:23333/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is PocketFlow?"}'
```

### 示例：上传文档并对话

```bash
# 上传文档
curl -X POST http://127.0.0.1:23333/upload \
  -H "Content-Type: application/json" \
  -d '{"texts": ["自定义文档内容"], "session_id": "my-session"}'

# 使用该 session 对话
curl -X POST http://127.0.0.1:23333/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "文档内容是什么？", "session_id": "my-session"}'
```

## 环境变量

| 变量名 | 说明 | 必填 |
|--------|------|------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | 是 |
| `HF_HUB_CACHE` | Hugging Face 模型缓存路径 | 否 |
| `BGE_MODEL_REPO` | BGE 模型仓库名 | 否 |

## 技术栈

- **PocketFlow**：100 行极简 LLM 框架
- **FastAPI**：Web 服务框架
- **FAISS**：向量相似度搜索
- **Sentence Transformers**：文本向量化
- **OpenAI**：LLM 调用

## 版本更新

### v1.1 (2026-06-27)

- 新增会话管理：支持多会话隔离，每个 session 独立文档和索引
- 新增 `/upload` 接口：支持上传自定义文档
- 新增 Session 中间件：统一处理 session 逻辑
- 重构项目结构：分离 common、sessions、rag_logic 模块
- 修复 shared 数据结构不一致问题
- 简化 `/chat` 接口逻辑

### v1.0 (2026-05-16)

- 初始版本
- 基础 RAG 功能：离线索引构建 + 在线查询
- FastAPI 服务接口

## License

MIT
