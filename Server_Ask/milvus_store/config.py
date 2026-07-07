"""Milvus 存储配置"""

# Milvus 连接配置
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530

# 集合配置
COLLECTION_NAME = "rag_documents"
VECTOR_DIM = 512  # BGE-small-zh 输出维度

# 索引配置
INDEX_PARAMS = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 128}
}

# 搜索配置
SEARCH_PARAMS = {
    "metric_type": "L2",
    "params": {"nprobe": 16}
}