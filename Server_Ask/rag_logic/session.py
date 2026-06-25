"""
内存会话管理：每个 session 维护独立的对话历史和文档索引。
"""
import uuid
import threading
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import faiss

from .utils import get_embeddings_2, fix_size_chunk


@dataclass
class SessionData:
    session_id: str
    history: list = field(default_factory=list)
    documents: list = field(default_factory=list)
    chunks: list = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    index: Optional[faiss.Index] = None


class SessionManager:
    def __init__(self):
        self._sessions: dict[str, SessionData] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData(session_id=session_id)
            return self._sessions[session_id]

    def add_history(self, session_id: str, query: str, answer: str):
        session = self.get_or_create(session_id)
        with self._lock:
            session.history.append({"query": query, "answer": answer})

    def get_history(self, session_id: str) -> list:
        session = self.get_or_create(session_id)
        return list(session.history)

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


session_manager = SessionManager()
