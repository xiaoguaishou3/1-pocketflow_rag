"""Layer 1: 纯函数测试 - utils.py"""
import sys
import os
import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Server_Ask"))

from common.utils import fix_size_chunk


class TestFixSizeChunk:
    """fix_size_chunk 函数测试"""

    def test_basic_chunking(self):
        """基本切分：3000 字符按 2000 切分"""
        text = "a" * 3000
        chunks = fix_size_chunk(text, chunk_size=2000)
        assert len(chunks) == 2
        assert chunks[0] == "a" * 2000
        assert chunks[1] == "a" * 1000

    def test_empty_text(self):
        """空文本返回空列表"""
        chunks = fix_size_chunk("")
        assert chunks == []

    def test_exact_size(self):
        """恰好等于 chunk_size"""
        text = "a" * 2000
        chunks = fix_size_chunk(text, chunk_size=2000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_small_text(self):
        """文本小于 chunk_size"""
        text = "hello"
        chunks = fix_size_chunk(text, chunk_size=2000)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_custom_chunk_size(self):
        """自定义 chunk_size"""
        text = "a" * 100
        chunks = fix_size_chunk(text, chunk_size=30)
        assert len(chunks) == 4
        assert chunks[0] == "a" * 30
        assert chunks[-1] == "a" * 10

    def test_single_char_chunks(self):
        """chunk_size=1，每个字符一个 chunk"""
        text = "abc"
        chunks = fix_size_chunk(text, chunk_size=1)
        assert chunks == ["a", "b", "c"]

    def test_unicode_text(self):
        """Unicode 文本切分"""
        text = "你好世界" * 100  # 400 字符
        chunks = fix_size_chunk(text, chunk_size=200)
        assert len(chunks) == 2
        assert all(len(c) == 200 for c in chunks)
