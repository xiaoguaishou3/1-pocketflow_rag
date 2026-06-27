"""
会话中间件：处理 session 逻辑，预构建 shared 存入 request.state。
"""
import uuid
import numpy as np
import faiss
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from sessions.session import session_manager
from common.utils import get_embeddings_2, fix_size_chunk
from common.defaults import DEFAULT_DOCUMENTS
from rag_logic.flow import online_flow


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/chat" and request.method == "POST":
            try:
                body = await request.json()
                query = body.get("query", "")
                session_id = body.get("session_id") or str(uuid.uuid4())

                if session_manager.has_documents(session_id):
                    chunks = session_manager.get_chunks(session_id)
                else:
                    chunks = self._get_default_chunks()

                shared = self._build_shared(query, chunks)
                request.state.shared = shared
                request.state.session_id = session_id
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Request parsing failed: {str(e)}"}
                )

        response = await call_next(request)
        return response

    def _get_default_chunks(self) -> list[str]:
        chunks = []
        for doc in DEFAULT_DOCUMENTS:
            chunks.extend(fix_size_chunk(doc))
        return chunks

    def _build_shared(self, query: str, chunks: list[str]) -> dict:
        all_embs = [get_embeddings_2(c) for c in chunks]
        emb_np = np.array(all_embs, dtype=np.float32)
        query_emb = np.array([get_embeddings_2(query)], dtype=np.float32)

        dim = emb_np.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(emb_np)
        dists, idces = idx.search(query_emb, k=1)
        best_idx = idces[0][0]

        return {
            "texts": chunks,
            "query": query,
            "query_embedding": query_emb,
            "retrieved_document": {
                "text": chunks[best_idx],
                "index": best_idx,
                "distance": float(dists[0][0])
            },
            "generated_answer": None
        }
