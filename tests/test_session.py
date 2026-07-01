import time
import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from typing import Optional, List

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Server_Ask"))

from sessions.session import SessionManager, SessionData, session_manager


class TestSessionManagerInit:
    """Test SessionManager.__init__"""

    def test_default_ttl(self):
        """Test default ttl_seconds value"""
        sm = SessionManager()
        assert sm._ttl_seconds == 86400
        assert isinstance(sm._ttl_seconds, int)

    def test_custom_ttl(self):
        """Test custom ttl_seconds value"""
        sm = SessionManager(ttl_seconds=3600)
        assert sm._ttl_seconds == 3600
        assert isinstance(sm._ttl_seconds, int)

    def test_init_empty_sessions(self):
        """Test that sessions dict is empty on init"""
        sm = SessionManager()
        assert sm._sessions == {}
        assert isinstance(sm._sessions, dict)

    def test_init_lock(self):
        """Test that lock is created"""
        sm = SessionManager()
        assert hasattr(sm, '_lock')


class TestSessionManagerGetOrCreate:
    """Test SessionManager.get_or_create"""

    def test_create_new_session(self):
        """Test creating a new session"""
        sm = SessionManager()
        session_id = "test_session_1"
        session = sm.get_or_create(session_id)
        
        assert isinstance(session, SessionData)
        assert session.session_id == session_id
        assert isinstance(session.created_at, float)
        assert session.shared == {}
        assert session_id in sm._sessions

    def test_get_existing_session(self):
        """Test getting an existing session"""
        sm = SessionManager()
        session_id = "test_session_2"
        
        # First call creates
        session1 = sm.get_or_create(session_id)
        # Second call returns existing
        session2 = sm.get_or_create(session_id)
        
        assert session1 is session2  # Same object
        assert sm._sessions[session_id] is session1

    def test_multiple_sessions(self):
        """Test multiple different sessions"""
        sm = SessionManager()
        session1 = sm.get_or_create("session_a")
        session2 = sm.get_or_create("session_b")
        
        assert session1.session_id == "session_a"
        assert session2.session_id == "session_b"
        assert session1 is not session2
        assert len(sm._sessions) == 2

    def test_empty_string_session_id(self):
        """Test with empty string as session_id"""
        sm = SessionManager()
        session = sm.get_or_create("")
        assert session.session_id == ""
        assert "" in sm._sessions


class TestSessionManagerAddDocuments:
    """Test SessionManager.add_documents"""

    @patch('sessions.session.offline_flow')
    def test_add_documents_normal(self, mock_flow):
        """Test adding documents normally"""
        mock_flow.run = MagicMock()
        sm = SessionManager()
        session_id = "test_add"
        texts = ["document1", "document2", "document3"]
        
        sm.add_documents(session_id, texts)
        
        # Verify flow was called
        mock_flow.run.assert_called_once()
        
        # Verify session exists and has shared data
        session = sm._sessions[session_id]
        assert session.session_id == session_id
        assert session.shared["texts"] == texts
        assert session.shared["embeddings"] is None
        assert session.shared["query"] is None

    @patch('sessions.session.offline_flow')
    def test_add_documents_empty_list(self, mock_flow):
        """Test adding empty document list"""
        mock_flow.run = MagicMock()
        sm = SessionManager()
        session_id = "test_empty"
        texts = []
        
        sm.add_documents(session_id, texts)
        
        mock_flow.run.assert_called_once()
        session = sm._sessions[session_id]
        assert session.shared["texts"] == []

    @patch('sessions.session.offline_flow')
    def test_add_documents_overwrite(self, mock_flow):
        """Test adding documents overwrites existing"""
        mock_flow.run = MagicMock()
        sm = SessionManager()
        session_id = "test_overwrite"
        
        # First add
        sm.add_documents(session_id, ["doc1"])
        first_shared = sm._sessions[session_id].shared.copy()
        
        # Second add
        sm.add_documents(session_id, ["doc2", "doc3"])
        second_shared = sm._sessions[session_id].shared
        
        assert second_shared["texts"] == ["doc2", "doc3"]
        assert mock_flow.run.call_count == 2

    @patch('sessions.session.offline_flow')
    def test_add_documents_flow_updates_shared(self, mock_flow):
        """Test that offline_flow updates shared correctly"""
        def mock_run(shared):
            shared["index"] = "some_index"
            shared["embeddings"] = [0.1, 0.2, 0.3]
        
        mock_flow.run = MagicMock(side_effect=mock_run)
        sm = SessionManager()
        session_id = "test_flow_updates"
        
        sm.add_documents(session_id, ["doc1"])
        session = sm._sessions[session_id]
        
        assert session.shared["index"] == "some_index"
        assert session.shared["embeddings"] == [0.1, 0.2, 0.3]


class TestSessionManagerGetShared:
    """Test SessionManager.get_shared"""

    @patch('sessions.session.offline_flow')
    def test_get_shared_with_data(self, mock_flow):
        """Test getting shared data after adding documents"""
        mock_flow.run = MagicMock()
        sm = SessionManager()
        session_id = "test_get_shared"
        texts = ["doc1"]
        
        sm.add_documents(session_id, texts)
        shared = sm.get_shared(session_id)
        
        assert shared is not None
        assert isinstance(shared, dict)
        assert shared["texts"] == texts
        assert shared is not sm._sessions[session_id].shared  # Different object

    def test_get_shared_no_session(self):
        """Test getting shared for non-existent session"""
        sm = SessionManager()
        shared = sm.get_shared("non_existent")
        assert shared is None

    @patch('sessions.session.offline_flow')
    def test_get_shared_empty_shared(self, mock_flow):
        """Test getting shared when session exists but no documents added"""
        mock_flow.run = MagicMock()
        sm = SessionManager()
        
        # Create session without adding documents
        sm.get_or_create("empty_session")
        shared = sm.get_shared("empty_session")
        assert shared is None  # shared is empty dict, so returns None

    @patch('sessions.session.offline_flow')
    def test_get_shared_returns_copy(self, mock_flow):
        """Test that get_shared returns a copy, not the original"""
        def mock_run(shared):
            shared["index"] = "test_index"
        
        mock_flow.run = MagicMock(side_effect=mock_run)
        sm = SessionManager()
        session_id = "test_copy"
        
        sm.add_documents(session_id, ["doc1"])
        shared = sm.get_shared(session_id)
        
        # Modify returned dict
        shared["new_key"] = "new_value"
        # Original should not be affected
        assert "new_key" not in sm._sessions[session_id].shared


class TestSessionManagerHasDocuments:
    """Test SessionManager.has_documents"""

    @patch('sessions.session.offline_flow')
    def test_has_documents_true(self, mock_flow):
        """Test has_documents returns True when index exists"""
        def mock_run(shared):
            shared["index"] = "some_index"
        
        mock_flow.run = MagicMock(side_effect=mock_run)
        sm = SessionManager()
        session_id = "test_has_docs"
        
        sm.add_documents(session_id, ["doc1"])
        assert sm.has_documents(session_id) is True

    @patch('sessions.session.offline_flow')
    def test_has_documents_false_no_index(self, mock_flow):
        """Test has_documents returns False when index is None"""
        mock_flow.run = MagicMock()
        sm = SessionManager()
        session_id = "test_no_index"
        
        sm.add_documents(session_id, ["doc1"])
        assert sm.has_documents(session_id) is False

    def test_has_documents_no_session(self):
        """Test has_documents for non-existent session"""
        sm = SessionManager()
        assert sm.has_documents("non_existent") is False

    @patch('sessions.session.offline_flow')
    def test_has_documents_session_no_shared(self, mock_flow):
        """Test has_documents when session exists but no shared data"""
        mock_flow.run = MagicMock()
        sm = SessionManager()
        
        sm.get_or_create("no_shared_session")
        assert sm.has_documents("no_shared_session") is False


class TestSessionManagerCleanupExpired:
    """Test SessionManager.cleanup_expired"""

    @patch('time.time')
    def test_cleanup_expired_sessions(self, mock_time):
        """Test cleanup removes expired sessions"""
        mock_time.return_value = 1000.0
        
        sm = SessionManager(ttl_seconds=100)
        
        # Create sessions
        session1 = sm.get_or_create("session1")
        session2 = sm.get_or_create("session2")
        
        # Manually set created_at to simulate expired session
        sm._sessions["session1"].created_at = 800.0  # 200 seconds ago (expired)
        sm._sessions["session2"].created_at = 950.0  # 50 seconds ago (not expired)
        
        mock_time.return_value = 1000.0
        
        sm.cleanup_expired()
        
        assert "session1" not in sm._sessions
        assert "session2" in sm._sessions

    @patch('time.time')
    def test_cleanup_no_expired_sessions(self, mock_time):
        """Test cleanup when no sessions are expired"""
        mock_time.return_value = 100.0
        
        sm = SessionManager(ttl_seconds=1000)
        
        # Create sessions
        sm.get_or_create("session1")
        sm.get_or_create("session2")
        
        # All sessions created at time 100, TTL is 1000, so none expired
        sm.cleanup_expired()
        
        assert len(sm._sessions) == 2
        assert "session1" in sm._sessions
        assert "session2" in sm._sessions

    @patch('time.time')
    def test_cleanup_all_expired(self, mock_time):
        """Test cleanup when all sessions are expired"""
        mock_time.return_value = 1000.0
        
        sm = SessionManager(ttl_seconds=50)
        
        # Create sessions with old timestamps
        sm.get_or_create("session1")
        sm.get_or_create("session2")
        
        # Set all created_at to expired times
        for session in sm._sessions.values():
            session.created_at = 100.0  # 900 seconds ago (expired)
        
        sm.cleanup_expired()
        
        assert len(sm._sessions) == 0

    def test_cleanup_empty_manager(self):
        """Test cleanup when session manager is empty"""
        sm = SessionManager()
        sm.cleanup_expired()  # Should not raise any exception
        assert len(sm._sessions) == 0