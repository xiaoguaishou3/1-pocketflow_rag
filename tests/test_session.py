"""Layer 1: 会话管理器测试"""
import time
from unittest.mock import patch, MagicMock
from sessions.session import SessionManager, SessionData


class TestSessionManager:
    """SessionManager 类测试"""

    def setup_method(self):
        """每个测试前创建新的 SessionManager"""
        self.manager = SessionManager(ttl_seconds=3600)

    def test_get_or_create_new(self):
        """创建新会话"""
        session = self.manager.get_or_create("test-123")
        assert session.session_id == "test-123"
        assert isinstance(session.shared, dict)
        assert session.created_at > 0

    def test_get_or_create_existing(self):
        """获取已存在的会话"""
        session1 = self.manager.get_or_create("test-123")
        session2 = self.manager.get_or_create("test-123")
        assert session1 is session2

    def test_has_documents_false(self):
        """新会话没有文档"""
        session = self.manager.get_or_create("test-123")
        assert self.manager.has_documents("test-123") == False

    def test_has_documents_true(self):
        """添加文档后 has_documents 返回 True"""
        session = self.manager.get_or_create("test-123")
        session.shared["index"] = MagicMock()  # 模拟 FAISS index
        assert self.manager.has_documents("test-123") == True

    def test_has_documents_nonexistent(self):
        """不存在的会话返回 False"""
        assert self.manager.has_documents("nonexistent") == False

    def test_get_shared(self):
        """获取 shared 字典"""
        session = self.manager.get_or_create("test-123")
        session.shared["key"] = "value"
        shared = self.manager.get_shared("test-123")
        assert shared["key"] == "value"

    def test_get_shared_nonexistent(self):
        """不存在的会话返回 None"""
        assert self.manager.get_shared("nonexistent") is None

    def test_cleanup_expired(self):
        """清理过期会话"""
        session = self.manager.get_or_create("test-123")
        # 手动设置过期时间
        session.created_at = time.time() - 7200  # 2 小时前
        self.manager.cleanup_expired()
        assert "test-123" not in self.manager._sessions

    def test_cleanup_not_expired(self):
        """未过期会话不被清理"""
        session = self.manager.get_or_create("test-123")
        session.created_at = time.time()  # 刚创建
        self.manager.cleanup_expired()
        assert "test-123" in self.manager._sessions

    def test_add_documents_mock(self):
        """添加文档（mock offline_flow）"""
        with patch("sessions.session.offline_flow") as mock_flow:
            mock_flow.run = MagicMock()
            session = self.manager.get_or_create("test-123")
            self.manager.add_documents("test-123", ["文档1", "文档2"])
            assert session.shared["texts"] == ["文档1", "文档2"]
            mock_flow.run.assert_called_once()
