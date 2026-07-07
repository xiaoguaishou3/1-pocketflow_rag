"""Milvus 集合管理"""

from pymilvus import Collection, FieldSchema, CollectionSchema, DataType, utility
from .config import COLLECTION_NAME, VECTOR_DIM, INDEX_PARAMS


def create_collection(name: str = COLLECTION_NAME, dim: int = VECTOR_DIM) -> Collection:
    """创建 Milvus 集合"""
    if utility.has_collection(name):
        return Collection(name)

    # 定义字段
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="session_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
    ]

    # 创建集合
    schema = CollectionSchema(fields=fields, description="RAG documents collection")
    collection = Collection(name=name, schema=schema)

    # 创建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    collection.create_index(field_name="embedding", index_params=index_params)

    return collection


def get_collection(name: str = COLLECTION_NAME) -> Collection:
    """获取集合，如果不存在则创建"""
    if not utility.has_collection(name):
        return create_collection(name)
    return Collection(name)


def drop_collection(name: str = COLLECTION_NAME) -> bool:
    """删除集合"""
    if utility.has_collection(name):
        utility.drop_collection(name)
        return True
    return False


def has_collection(name: str = COLLECTION_NAME) -> bool:
    """检查集合是否存在"""
    return utility.has_collection(name)