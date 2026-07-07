"""Milvus 连接管理"""

from pymilvus import connections, utility
from .config import MILVUS_HOST, MILVUS_PORT


def connect(host: str = MILVUS_HOST, port: str = MILVUS_PORT) -> None:
    """连接到 Milvus 服务器"""
    connections.connect(
        alias="default",
        host=host,
        port=port
    )


def disconnect() -> None:
    """断开 Milvus 连接"""
    connections.disconnect("default")


def get_server_version() -> str:
    """获取 Milvus 服务器版本"""
    return utility.get_server_version()


def is_connected() -> bool:
    """检查是否已连接"""
    try:
        utility.get_server_version()
        return True
    except Exception:
        return False