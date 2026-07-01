"""共享测试 fixtures"""
import sys
import os
import pytest
import numpy as np

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Server_Ask"))


@pytest.fixture
def sample_texts():
    """示例文档文本"""
    return [
        "PocketFlow 是一个极简的 LLM 框架",
        "RAG 是检索增强生成的缩写",
        "FAISS 用于高效的向量相似度搜索"
    ]


@pytest.fixture
def sample_query():
    """示例查询"""
    return "什么是 PocketFlow？"


@pytest.fixture
def sample_shared():
    """示例 shared 字典"""
    return {
        "query": "什么是 PocketFlow？",
        "texts": [],
        "embeddings": None,
        "index": None
    }


@pytest.fixture
def mock_embedding():
    """模拟的 embedding 向量"""
    return np.random.rand(512).astype(np.float32)
