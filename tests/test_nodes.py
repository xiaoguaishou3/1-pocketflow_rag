import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock, ANY

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Server_Ask"))

from rag_logic.nodes import (
    ChunkDocumentsNode,
    EmbedDocumentsNode,
    CreateIndexNode,
    EmbedQueryNode,
    RetrieveDocumentNode,
    GenerateAnswerNode,
)
from common.utils import fix_size_chunk, get_embeddings_2, call_llm


class TestChunkDocumentsNode:
    def setup_method(self):
        self.node = ChunkDocumentsNode()

    def test_prep_returns_texts(self):
        shared = {"texts": ["doc1", "doc2", "doc3"]}
        result = self.node.prep(shared)
        assert result == ["doc1", "doc2", "doc3"]
        assert isinstance(result, list)

    def test_prep_empty_list(self):
        shared = {"texts": []}
        result = self.node.prep(shared)
        assert result == []
        assert isinstance(result, list)

    def test_exec_normal_text(self):
        text = "This is a sample text for testing chunking functionality."
        result = self.node.exec(text)
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(chunk, str) for chunk in result)

    def test_exec_empty_string(self):
        text = ""
        result = self.node.exec(text)
        assert isinstance(result, list)
        assert len(result) == 0

    def test_exec_very_long_text(self):
        text = "A" * 10000
        result = self.node.exec(text)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_post_updates_shared_and_returns_default(self):
        shared = {"texts": ["test"]}
        prep_res = ["doc1", "doc2"]
        exec_res_list = [["chunk1", "chunk2"], ["chunk3"]]
        result = self.node.post(shared, prep_res, exec_res_list)
        assert shared["texts"] == ["chunk1", "chunk2", "chunk3"]
        assert result == "default"

    def test_post_empty_exec_res_list(self):
        shared = {"texts": ["test"]}
        prep_res = ["doc1"]
        exec_res_list = [[]]
        result = self.node.post(shared, prep_res, exec_res_list)
        assert shared["texts"] == []
        assert result == "default"


class TestEmbedDocumentsNode:
    def setup_method(self):
        self.node = EmbedDocumentsNode()

    def test_prep_returns_texts(self):
        shared = {"texts": ["text1", "text2"]}
        result = self.node.prep(shared)
        assert result == ["text1", "text2"]
        assert isinstance(result, list)

    def test_prep_empty_list(self):
        shared = {"texts": []}
        result = self.node.prep(shared)
        assert result == []
        assert isinstance(result, list)

    def test_exec_normal_text(self):
        text = "Sample text for embedding"
        result = self.node.exec(text)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32

    def test_exec_empty_string(self):
        text = ""
        result = self.node.exec(text)
        assert isinstance(result, np.ndarray)

    def test_post_updates_shared_with_embeddings(self):
        shared = {"texts": ["text1", "text2"]}
        prep_res = ["text1", "text2"]
        mock_embeddings = [np.array([0.1, 0.2, 0.3], dtype=np.float32),
                          np.array([0.4, 0.5, 0.6], dtype=np.float32)]
        exec_res_list = mock_embeddings
        result = self.node.post(shared, prep_res, exec_res_list)
        assert "embeddings" in shared
        assert isinstance(shared["embeddings"], np.ndarray)
        assert shared["embeddings"].shape == (2, 3)
        assert shared["embeddings"].dtype == np.float32
        assert result == "default"

    def test_post_empty_exec_res_list(self):
        shared = {"texts": []}
        prep_res = []
        exec_res_list = []
        result = self.node.post(shared, prep_res, exec_res_list)
        assert "embeddings" in shared
        assert isinstance(shared["embeddings"], np.ndarray)
        assert shared["embeddings"].shape[0] == 0
        assert result == "default"


class TestCreateIndexNode:
    def setup_method(self):
        self.node = CreateIndexNode()

    def test_prep_returns_embeddings(self):
        shared = {"embeddings": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)}
        result = self.node.prep(shared)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 2)

    def test_prep_single_embedding(self):
        shared = {"embeddings": np.array([[0.5, 0.6, 0.7]], dtype=np.float32)}
        result = self.node.prep(shared)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1, 3)

    def test_exec_creates_valid_index(self):
        embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        result = self.node.exec(embeddings)
        import faiss
        assert isinstance(result, faiss.IndexFlatL2)
        assert result.ntotal == 3
        assert result.d == 2

    def test_exec_single_embedding(self):
        embeddings = np.array([[1.0, 2.0]], dtype=np.float32)
        result = self.node.exec(embeddings)
        assert result.ntotal == 1
        assert result.d == 2

    def test_post_updates_shared(self):
        shared = {"embeddings": np.array([[0.1, 0.2]], dtype=np.float32)}
        mock_index = MagicMock()
        mock_index.ntotal = 1
        result = self.node.post(shared, None, mock_index)
        assert shared["index"] == mock_index
        assert result == "default"


class TestEmbedQueryNode:
    def setup_method(self):
        self.node = EmbedQueryNode()

    def test_prep_returns_query(self):
        shared = {"query": "What is AI?"}
        result = self.node.prep(shared)
        assert result == "What is AI?"
        assert isinstance(result, str)

    def test_prep_empty_query(self):
        shared = {"query": ""}
        result = self.node.prep(shared)
        assert result == ""

    def test_exec_normal_query(self):
        query = "Test query for embedding"
        result = self.node.exec(query)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert result.shape[0] == 1

    def test_exec_empty_query(self):
        query = ""
        result = self.node.exec(query)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1

    def test_post_updates_shared(self):
        shared = {"query": "test"}
        mock_embedding = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        result = self.node.post(shared, None, mock_embedding)
        assert np.array_equal(shared["query_embedding"], mock_embedding)
        assert result == "default"


class TestRetrieveDocumentNode:
    def setup_method(self):
        self.node = RetrieveDocumentNode()

    def test_prep_returns_tuple(self):
        shared = {
            "query_embedding": np.array([[0.1, 0.2]], dtype=np.float32),
            "index": MagicMock(),
            "texts": ["doc1", "doc2", "doc3"]
        }
        q_emb, idx, texts = self.node.prep(shared)
        assert np.array_equal(q_emb, np.array([[0.1, 0.2]], dtype=np.float32))
        assert idx == shared["index"]
        assert texts == ["doc1", "doc2", "doc3"]

    def test_prep_single_text(self):
        shared = {
            "query_embedding": np.array([[0.5]], dtype=np.float32),
            "index": MagicMock(),
            "texts": ["only_doc"]
        }
        q_emb, idx, texts = self.node.prep(shared)
        assert texts == ["only_doc"]

    @patch('rag_logic.nodes.faiss')
    def test_exec_returns_correct_dict(self, mock_faiss):
        mock_index = MagicMock()
        mock_index.search.return_value = (np.array([[0.5]]), np.array([[2]]))
        query_embedding = np.array([[0.1, 0.2]], dtype=np.float32)
        texts = ["doc0", "doc1", "doc2"]
        inputs = (query_embedding, mock_index, texts)
        result = self.node.exec(inputs)
        assert isinstance(result, dict)
        assert "text" in result
        assert "index" in result
        assert "distance" in result
        assert result["text"] == "doc2"
        assert result["index"] == 2
        assert result["distance"] == 0.5

    def test_post_updates_shared(self):
        shared = {
            "query_embedding": np.array([[0.1]], dtype=np.float32),
            "index": MagicMock(),
            "texts": ["doc1"]
        }
        retrieved_doc = {"text": "doc1", "index": 0, "distance": 0.1}
        result = self.node.post(shared, None, retrieved_doc)
        assert shared["retrieved_document"] == retrieved_doc
        assert result == "default"


class TestGenerateAnswerNode:
    def setup_method(self):
        self.node = GenerateAnswerNode()

    def test_prep_returns_tuple(self):
        shared = {
            "query": "What is Python?",
            "retrieved_document": {"text": "Python is a language", "index": 0, "distance": 0.5}
        }
        query, retrieved_doc = self.node.prep(shared)
        assert query == "What is Python?"
        assert retrieved_doc == shared["retrieved_document"]

    def test_prep_empty_query(self):
        shared = {
            "query": "",
            "retrieved_document": {"text": "Some context", "index": 0, "distance": 0.3}
        }
        query, retrieved_doc = self.node.prep(shared)
        assert query == ""

    @patch('rag_logic.nodes.call_llm')
    def test_exec_returns_answer(self, mock_call_llm):
        mock_call_llm.return_value = "Python is a programming language."
        inputs = ("What is Python?", {"text": "Python is a language", "index": 0, "distance": 0.5})
        result = self.node.exec(inputs)
        assert isinstance(result, str)
        assert result == "Python is a programming language."
        mock_call_llm.assert_called_once()

    @patch('rag_logic.nodes.call_llm')
    def test_exec_empty_context(self, mock_call_llm):
        mock_call_llm.return_value = ""
        inputs = ("Question?", {"text": "", "index": 0, "distance": 0.5})
        result = self.node.exec(inputs)
        assert isinstance(result, str)
        assert result == ""

    def test_post_updates_shared(self):
        shared = {
            "query": "test",
            "retrieved_document": {"text": "context", "index": 0, "distance": 0.5}
        }
        mock_answer = "This is a generated answer."
        result = self.node.post(shared, None, mock_answer)
        assert shared["generated_answer"] == mock_answer
        assert result == "default"