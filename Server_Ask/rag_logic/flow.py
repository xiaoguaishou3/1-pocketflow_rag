"""
@File   : flow
@Author : 74775
@Date   : 2026/5/16 21:30
Define offline and online PocketFlow workflows.
Offline: Document chunking → Embedding → FAISS index (runs once at startup)
Online: Query embedding → Retrieval → LLM generation (runs per request)
"""
from pocketflow import Flow
from .nodes import ChunkDocumentsNode, EmbedDocumentsNode, CreateIndexNode, EmbedQueryNode, RetrieveDocumentNode, GenerateAnswerNode

def get_offline_flow():
    """Offline workflow: chunking → vectorization → index building"""
    chunk_docs_node = ChunkDocumentsNode()
    embed_docs_node = EmbedDocumentsNode()
    create_index_node = CreateIndexNode()

    # >> 运算符串联节点：前一个节点的 post 结果作为下一个节点的 prep 输入
    chunk_docs_node >> embed_docs_node >> create_index_node

    return Flow(start=chunk_docs_node)


def get_online_flow():
    """Online workflow: query vectorization → document retrieval → answer generation"""
    embed_query_node = EmbedQueryNode()
    retrieve_doc_node = RetrieveDocumentNode()
    generate_answer_node = GenerateAnswerNode()

    embed_query_node >> retrieve_doc_node >> generate_answer_node

    return Flow(start=embed_query_node)


offline_flow = get_offline_flow()
online_flow = get_online_flow()
