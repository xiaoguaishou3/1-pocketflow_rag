"""
内存会话管理：每个 session 存储 offline_flow 生成的 shared 对象。
"""
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from common.defaults import DEFAULT_DOCUMENTS
from rag_logic.flow import offline_flow


@dataclass
class SessionData:
    session_id: str
    shared: dict = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


class SessionManager:
    def __init__(self, ttl_seconds: int = 86400):
        self._sessions: dict[str, SessionData] = {}
        self._lock = threading.Lock()
        self._ttl_seconds = ttl_seconds

    def get_or_create(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(session_id=session_id)
            return self._sessions[session_id]

    def add_documents(self, session_id: str, texts: list[str]):
        session = self.get_or_create(session_id)
        with self._lock:
            shared = {
                "texts": list(texts),
                "embeddings": None,
                "query": None,
                "query_embedding": None,
                "retrieved_document": None,
                "generated_answer": None
            }
            offline_flow.run(shared)
            session.shared = shared

    def get_shared(self, session_id: str) -> Optional[dict]:
        session = self._sessions.get(session_id)
        if session and session.shared:
            return dict(session.shared)
        return None

    def has_documents(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session and session.shared:
            return session.shared.get("index") is not None
        return False

    def cleanup_expired(self):
        with self._lock:
            current_time = time.time()
            expired = [
                sid for sid, session in self._sessions.items()
                if current_time - session.created_at > self._ttl_seconds
            ]
            for sid in expired:
                del self._sessions[sid]


session_manager = SessionManager()
