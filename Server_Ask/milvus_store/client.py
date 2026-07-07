"""Milvus 连接管理"""

from pymilvus import MilvusClient
from .config import MILVUS_HOST, MILVUS_PORT


def connect(host: str = MILVUS_HOST, port: str = MILVUS_PORT):
    """connect Milvus server"""
    client = MilvusClient(
        alias="default",
        host=host,
        port=port
    )

    return client


def get_milvus_version() -> str:
    """get Milvus version"""
    m_client = connect()
    return m_client.get_server_version()


def is_connected() -> bool:
    """check connection"""
    try:
        get_milvus_version()
        return True
    except Exception:
        return False