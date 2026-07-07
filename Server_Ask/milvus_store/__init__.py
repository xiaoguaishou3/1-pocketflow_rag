"""Milvus 存储包"""

from .client import connect, disconnect, get_server_version, is_connected
from .collection import create_collection, get_collection, drop_collection, has_collection
from .crud import insert_vectors, search_vectors, get_vector_by_id, delete_vectors, count_vectors
from .config import MILVUS_HOST, MILVUS_PORT, COLLECTION_NAME, VECTOR_DIM

__all__ = [
    # 客户端
    "connect",
    "disconnect",
    "get_server_version",
    "is_connected",
    # 集合管理
    "create_collection",
    "get_collection",
    "drop_collection",
    "has_collection",
    # CRUD 操作
    "insert_vectors",
    "search_vectors",
    "get_vector_by_id",
    "delete_vectors",
    "count_vectors",
    # 配置
    "MILVUS_HOST",
    "MILVUS_PORT",
    "COLLECTION_NAME",
    "VECTOR_DIM",
]