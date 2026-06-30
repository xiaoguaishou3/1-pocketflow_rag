"""Layer 2: PocketFlow 节点测试"""
import numpy as np
from unittest.mock import patch, MagicMock
from rag_logic.nodes import (
    ChunkDocumentsNode,
    EmbedDocumentsNode,
    CreateIndexNode,
    EmbedQueryNode,
    RetrieveDocumentNode,
    GenerateAnswerNode,
)


class TestChunkDocumentsNode:
    """ChunkDocumentsNode 节点测试"""

    def test_prep(self):
        """prep 返回 texts"""
        shared = {"texts": ["文档1", "文档2"]}
        node = ChunkDocumentsNode()
        result = node.prep(shared)
        assert result == ["文档1", "文档2"]

    def test_exec(self):
        """exec 切分文本"""
        node = ChunkDocumentsNode()
        chunks = node.exec("测试文本" * 100)
        assert isinstance(chunks, list)
        assert len(chunks) > 0

    def test_post(self):
        """post 更新 shared 并返回 default"""
        shared = {"texts": ["长文本" * 1000]}
        node = ChunkDocumentsNode()
        prep_res = ["长文本" * 1000]
        exec_res_list = [["chunk1", "chunk2"]]
        result = node.post(shared, prep_res, exec_res_list)
        assert result == "default"
        assert shared["texts"] == ["chunk1", "chunk2"]


class TestEmbedDocumentsNode:
    """EmbedDocumentsNode 节点测试"""

    def test_prep(self):
        """prep 返回 texts"""
        shared = {"texts": ["文本1", "文本2"]}
        node = EmbedDocumentsNode()
        result = node.prep(shared)
        assert result == ["文本1", "文本2"]

    @patch("rag_logic.nodes.get_embeddings_2")
    def test_exec(self, mock_embeddings):
        """exec 生成 embeddings"""
        mock_embeddings.return_value = np.random.rand(512).astype(np.float32)
        node = EmbedDocumentsNode()
        result = node.exec("测试文本")
        assert isinstance(result, np.ndarray)

    def test_post(self):
        """post 更新 shared"""
        shared = {"texts": ["文本1"]}
        node = EmbedDocumentsNode()
        exec_res_list = [np.random.rand(512).astype(np.float32)]
        result = node.post(shared, ["文本1"], exec_res_list)
        assert result == "default"
        assert shared["embeddings"] is not None


class TestCreateIndexNode:
    """CreateIndexNode 节点测试"""

    def test_prep(self):
        """prep 返回 embeddings"""
        shared = {"embeddings": np.random.rand(2, 512).astype(np.float32)}
        node = CreateIndexNode()
        result = node.prep(shared)
        assert isinstance(result, np.ndarray)

    def test_exec(self):
        """exec 创建 FAISS 索引"""
        node = CreateIndexNode()
        embeddings = np.random.rand(3, 512).astype(np.float32)
        index = node.exec(embeddings)
        assert index.ntotal == 3

    def test_post(self):
        """post 更新 shared"""
        shared = {"embeddings": np.random.rand(2, 512).astype(np.float32)}
        node = CreateIndexNode()
        mock_index = MagicMock()
        mock_index.ntotal = 2
        result = node.post(shared, shared["embeddings"], mock_index)
        assert result == "default"
        assert shared["index"] is mock_index


class TestEmbedQueryNode:
    """EmbedQueryNode 节点测试"""

    def test_prep(self):
        """prep 返回 query"""
        shared = {"query": "什么是 RAG？"}
        node = EmbedQueryNode()
        result = node.prep(shared)
        assert result == "什么是 RAG？"

    @patch("rag_logic.nodes.get_embeddings_2")
    def test_exec(self, mock_embeddings):
        """exec 嵌入查询"""
        mock_embeddings.return_value = np.random.rand(512).astype(np.float32)
        node = EmbedQueryNode()
        result = node.exec("测试查询")
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # 二维数组

    def test_post(self):
        """post 更新 shared"""
        shared = {"query": "测试"}
        node = EmbedQueryNode()
        exec_res = np.random.rand(1, 512).astype(np.float32)
        result = node.post(shared, "测试", exec_res)
        assert result == "default"
        assert "query_embedding" in shared


class TestRetrieveDocumentNode:
    """RetrieveDocumentNode 节点测试"""

    def test_prep(self):
        """prep 返回 tuple"""
        shared = {
            "query_embedding": np.random.rand(1, 512).astype(np.float32),
            "index": MagicMock(),
            "texts": ["文本1", "文本2"]
        }
        node = RetrieveDocumentNode()
        result = node.prep(shared)
        assert len(result) == 3

    def test_exec(self):
        """exec 检索文档"""
        node = RetrieveDocumentNode()
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.5]]),
            np.array([[0]])
        )
        inputs = (
            np.random.rand(1, 512).astype(np.float32),
            mock_index,
            ["文档1", "文档2"]
        )
        result = node.exec(inputs)
        assert result["text"] == "文档1"
        assert result["index"] == 0
        assert result["distance"] == 0.5

    def test_post(self):
        """post 更新 shared"""
        shared = {}
        node = RetrieveDocumentNode()
        exec_res = {"text": "文档", "index": 0, "distance": 0.5}
        result = node.post(shared, None, exec_res)
        assert result == "default"
        assert shared["retrieved_document"] == exec_res


class TestGenerateAnswerNode:
    """GenerateAnswerNode 节点测试"""

    def test_prep(self):
        """prep 返回 tuple"""
        shared = {
            "query": "什么是 RAG？",
            "retrieved_document": {"text": "RAG 是检索增强生成"}
        }
        node = GenerateAnswerNode()
        result = node.prep(shared)
        assert len(result) == 2

    @patch("rag_logic.nodes.call_llm")
    def test_exec(self, mock_llm):
        """exec 调用 LLM 生成答案"""
        mock_llm.return_value = "RAG 是检索增强生成技术"
        node = GenerateAnswerNode()
        inputs = ("什么是 RAG？", {"text": "RAG 是检索增强生成"})
        result = node.exec(inputs)
        assert isinstance(result, str)
        mock_llm.assert_called_once()

    def test_post(self):
        """post 更新 shared"""
        shared = {}
        node = GenerateAnswerNode()
        result = node.post(shared, None, "生成的答案")
        assert result == "default"
        assert shared["generated_answer"] == "生成的答案"
