"""
@File   : flow
@Author : 74775
@Date   : 2026/5/16 21:30
定义离线和在线两条 PocketFlow 流程。
离线：文档切片 → Embedding → FAISS 索引（启动时执行一次）
在线：查询 Embedding → 检索 → LLM 生成（每次请求执行）
"""
from pocketflow import Flow
from .nodes import ChunkDocumentsNode, EmbedDocumentsNode, CreateIndexNode, EmbedQueryNode, RetrieveDocumentNode, GenerateAnswerNode

def get_offline_flow():
    """离线流程：文档切片 → 向量化 → 建索引"""
    chunk_docs_node = ChunkDocumentsNode()
    embed_docs_node = EmbedDocumentsNode()
    create_index_node = CreateIndexNode()

    # >> 运算符串联节点：前一个节点的 post 结果作为下一个节点的 prep 输入
    chunk_docs_node >> embed_docs_node >> create_index_node

    return Flow(start=chunk_docs_node)


def get_online_flow():
    """在线流程：查询向量化 → 检索文档 → 生成回答"""
    embed_query_node = EmbedQueryNode()
    retrieve_doc_node = RetrieveDocumentNode()
    generate_answer_node = GenerateAnswerNode()

    embed_query_node >> retrieve_doc_node >> generate_answer_node

    return Flow(start=embed_query_node)


offline_flow = get_offline_flow()
online_flow = get_online_flow()
