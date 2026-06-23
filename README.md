# 1-pocketflow_rag

基于 [PocketFlow](https://github.com/The-Pocket/PocketFlow) 的 RAG（检索增强生成）示例项目。

## 项目结构

```
├── main.py                 # CLI 演示入口
├── requirement.txt         # 依赖列表
└── Server_Ask/
    ├── start_server.py     # FastAPI 服务入口
    ├── run_rag.py          # RAG 共享数据管理
    └── rag_logic/
        ├── flow.py         # 离线/在线流程定义
        ├── nodes.py        # PocketFlow 节点实现
        └── utils.py        # 工具函数（LLM 调用、Embedding）
```

## 功能

- **离线流程**：文档切片 → Embedding 向量化 → FAISS 索引构建
- **在线流程**：查询向量化 → 相似文档检索 → LLM 生成回答

## 快速开始

```bash
pip install -r requirement.txt
python main.py --你的问题
```

## 启动 API 服务

```bash
cd Server_Ask
python start_server.py
```

服务启动后访问 `http://127.0.0.1:23333/chat` 发送 POST 请求获取回答。
