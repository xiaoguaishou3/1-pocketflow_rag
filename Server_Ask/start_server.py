"""
FastAPI 服务入口：支持会话管理和文档上传的 RAG 服务。
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
import uvicorn
import uuid
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional

from run_rag import offline_get_shared_store, get_default_documents
from rag_logic.flow import offline_flow, online_flow
from rag_logic.session import session_manager
from rag_logic.utils import get_embeddings_2, fix_size_chunk

default_shared = {}


@asynccontextmanager
async def startup(app: FastAPI):
    global default_shared
    default_shared = offline_get_shared_store()
    offline_flow.run(default_shared)
    yield


app = FastAPI(lifespan=startup)


@app.get("/")
async def homepage():
    return {"message": "Hello, World!"}


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    history: list


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = session_manager.get_or_create(session_id)

    has_session_docs = session.index is not None and session.index.ntotal > 0

    if has_session_docs:
        query_emb = get_embeddings_2(req.query)
        query_embedding = [query_emb]
        import numpy as np
        query_embedding_np = np.array(query_embedding, dtype=np.float32)

        distances, indices = session_manager.search(session_id, query_embedding_np, k=1)
        chunks = session_manager.get_chunks(session_id)
        best_idx = indices[0][0]
        retrieved_text = chunks[best_idx]

        shared_to_online_flow = {
            "texts": chunks,
            "query": req.query,
            "query_embedding": query_embedding_np,
            "retrieved_document": {
                "text": retrieved_text,
                "index": best_idx,
                "distance": float(distances[0][0])
            },
            "generated_answer": None
        }
        online_flow.run(shared_to_online_flow)
        answer = shared_to_online_flow.get("generated_answer", "")
    else:
        from rag_logic.utils import call_llm
        default_docs = get_default_documents()
        all_chunks = []
        for doc in default_docs:
            all_chunks.extend(fix_size_chunk(doc))

        import numpy as np
        all_embs = [get_embeddings_2(c) for c in all_chunks]
        emb_np = np.array(all_embs, dtype=np.float32)
        query_emb = np.array([get_embeddings_2(req.query)], dtype=np.float32)

        import faiss
        dim = emb_np.shape[1]
        idx = faiss.IndexFlatL2(dim)
        idx.add(emb_np)
        dists, idces = idx.search(query_emb, k=1)
        best_idx = idces[0][0]
        retrieved_text = all_chunks[best_idx]

        shared_to_online_flow = {
            "texts": all_chunks,
            "query": req.query,
            "query_embedding": query_emb,
            "retrieved_document": {
                "text": retrieved_text,
                "index": best_idx,
                "distance": float(dists[0][0])
            },
            "generated_answer": None
        }
        online_flow.run(shared_to_online_flow)
        answer = shared_to_online_flow.get("generated_answer", "")

    session_manager.add_history(session_id, req.query, answer)
    history = session_manager.get_history(session_id)

    return ChatResponse(answer=answer, session_id=session_id, history=history)


class UploadRequest(BaseModel):
    texts: list[str]
    session_id: Optional[str] = None


class UploadResponse(BaseModel):
    session_id: str
    document_count: int
    chunk_count: int


@app.post("/upload", response_model=UploadResponse)
async def upload(req: UploadRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session_manager.add_documents(session_id, req.texts)

    session = session_manager.get_or_create(session_id)
    return UploadResponse(
        session_id=session_id,
        document_count=len(session.documents),
        chunk_count=len(session.chunks)
    )


class HistoryResponse(BaseModel):
    session_id: str
    history: list


@app.get("/history/{session_id}", response_model=HistoryResponse)
async def get_history(session_id: str):
    history = session_manager.get_history(session_id)
    return HistoryResponse(session_id=session_id, history=history)


if __name__ == "__main__":
    uvicorn.run("start_server:app", host="127.0.0.1", port=23333)
