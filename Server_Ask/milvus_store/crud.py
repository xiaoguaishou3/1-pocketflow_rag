"""Milvus CRUD 操作"""

from typing import List, Dict, Any, Optional
from pymilvus import Collection
from .collection import get_collection
from .config import COLLECTION_NAME, SEARCH_PARAMS


def insert_vectors(
    texts: List[str],
    embeddings: List[List[float]],
    session_id: str,
    chunk_indices: List[int],
    collection_name: str = COLLECTION_NAME
) -> List[int]:
    """插入向量数据"""
    collection = get_collection(collection_name)

    # 准备数据
    data = [
        texts,
        embeddings,
        [session_id] * len(texts),
        chunk_indices
    ]

    # 插入数据
    result = collection.insert(data)
    collection.flush()

    return result.primary_keys


def search_vectors(
    query_embedding: List[float],
    k: int = 5,
    session_id: Optional[str] = None,
    collection_name: str = COLLECTION_NAME
) -> List[Dict[str, Any]]:
    """搜索相似向量"""
    collection = get_collection(collection_name)
    collection.load()

    # 构建搜索参数
    search_params = {
        "metric_type": "L2",
        "params": {"nprobe": 16}
    }

    # 构建过滤条件
    expr = f'session_id == "{session_id}"' if session_id else None

    # 执行搜索
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=k,
        expr=expr,
        output_fields=["text", "session_id", "chunk_index"]
    )

    # 格式化结果
    formatted_results = []
    for hits in results:
        for hit in hits:
            formatted_results.append({
                "id": hit.id,
                "text": hit.entity.get("text"),
                "session_id": hit.entity.get("session_id"),
                "chunk_index": hit.entity.get("chunk_index"),
                "distance": hit.distance
            })

    return formatted_results


def get_vector_by_id(
    vector_id: int,
    collection_name: str = COLLECTION_NAME
) -> Optional[Dict[str, Any]]:
    """按 ID 获取向量"""
    collection = get_collection(collection_name)
    collection.load()

    results = collection.query(
        expr=f"id == {vector_id}",
        output_fields=["text", "embedding", "session_id", "chunk_index"]
    )

    if results:
        return {
            "id": results[0]["id"],
            "text": results[0]["text"],
            "embedding": results[0]["embedding"],
            "session_id": results[0]["session_id"],
            "chunk_index": results[0]["chunk_index"]
        }
    return None


def delete_vectors(
    vector_ids: List[int],
    collection_name: str = COLLECTION_NAME
) -> bool:
    """删除向量"""
    collection = get_collection(collection_name)

    # 构建删除表达式
    ids_str = ",".join(str(id) for id in vector_ids)
    expr = f"id in [{ids_str}]"

    collection.delete(expr)
    collection.flush()

    return True


def count_vectors(collection_name: str = COLLECTION_NAME) -> int:
    """统计向量数量"""
    collection = get_collection(collection_name)
    return collection.num_entities