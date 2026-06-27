"""
内存会话管理：每个 session 维护独立的文档和索引，24小时自动清理。
"""
import uuid
import threading
import time
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import faiss

from common.utils import get_embeddings_2, fix_size_chunk


@dataclass
class SessionData:
    session_id: str
    documents: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    index: Optional[faiss.Index] = None
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
            session.documents.extend(texts)
            new_chunks = []
            for text in texts:
                new_chunks.extend(fix_size_chunk(text))
            session.chunks.extend(new_chunks)

            all_embeddings = []
            for chunk in session.chunks:
                all_embeddings.append(get_embeddings_2(chunk))
            session.embeddings = np.array(all_embeddings, dtype=np.float32)

            dimension = session.embeddings.shape[1]
            session.index = faiss.IndexFlatL2(dimension)
            session.index.add(session.embeddings)

    def search(self, session_id: str, query_embedding: np.ndarray, k: int = 1):
        session = self.get_or_create(session_id)
        if session.index is None or session.index.ntotal == 0:
            return None, None
        distances, indices = session.index.search(query_embedding, k)
        return distances, indices

    def get_chunks(self, session_id: str) -> list:
        session = self.get_or_create(session_id)
        return session.chunks

    def has_documents(self, session_id: str) -> bool:
        session = self.get_or_create(session_id)
        return session.index is not None and session.index.ntotal > 0

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
