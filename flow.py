"""
@File   : flow
@Author : 74775
@Date   : 2026/5/16 21:30
"""
from pocketflow import Flow
from nodes import EmbedDocumentsNode, CreateIndexNode, EmbedQueryNode, RetrieveDocumentNode, ChunkDocumentsNode, GenerateAnswerNode

def get_offline_flow():
    chunk_docs_node = ChunkDocumentsNode()
    embed_docs_node = EmbedDocumentsNode()
    create_index_node = CreateIndexNode()

    # __rshift__ 是 PocketFlow 中的运算符，表示将左边的节点输出作为右边的节点的输入，基类中实现了__rshift__方法，所以可以这样写
    chunk_docs_node >> embed_docs_node >> create_index_node

    offline_flow = Flow(start=chunk_docs_node)
    return offline_flow

def get_online_flow():
    embed_query_node = EmbedQueryNode()
    retrieve_doc_node = RetrieveDocumentNode()
    generate_answer_node = GenerateAnswerNode()

    embed_query_node >> retrieve_doc_node >> generate_answer_node

    online_flow = Flow(start=embed_query_node)
    return online_flow

# Initialize the flow
offline_flow = get_offline_flow()
online_flow = get_online_flow()
