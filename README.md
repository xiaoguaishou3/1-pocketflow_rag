# 1-pocketflow_rag

基于 [PocketFlow](https://github.com/The-Pocket/PocketFlow) 的 RAG（检索增强生成）示例项目。

## 项目结构

```
├── main.py                 # CLI 演示入口
├── requirement.txt         # 依赖列表
└── Server_Ask/
    ├── start_server.py     # FastAPI 服务入口
    ├── run_rag.py          # RAG 共享数据管理
    ├── .env.example        # 环境变量配置示例
    └── rag_logic/
        ├── __init__.py     # 包初始化
        ├── flow.py         # 离线/在线流程定义
        ├── nodes.py        # PocketFlow 节点实现
        └── utils.py        # 工具函数（LLM 调用、Embedding）
```

## 功能

- **离线流程**：文档切片 → Embedding 向量化 → FAISS 索引构建
- **在线流程**：查询向量化 → 相似文档检索 → LLM 生成回答

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirement.txt
```

### 2. 配置环境变量

```bash
cd Server_Ask
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

### 3. 运行 CLI 演示

```bash
python main.py --你的问题
```

### 4. 启动 API 服务

```bash
cd Server_Ask
python start_server.py
```

服务启动后访问 `http://127.0.0.1:23333/chat` 发送 POST 请求获取回答。

## API 接口

| 方法 | 路径 | 说明 | 参数 |
|------|------|------|------|
| GET | `/` | 首页 | 无 |
| POST | `/chat` | 发送问题 | `query: str` |

### 示例请求

```bash
curl -X POST http://127.0.0.1:23333/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is PocketFlow?"}'
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

## License

MIT
