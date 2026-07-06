"""
FastAPI 服务入口：支持会话管理和文档上传的 RAG 服务。
"""
from fastapi import FastAPI, Request
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import Optional

from rag_logic.flow import offline_flow, online_flow
from sessions.session import session_manager
from middleware import SessionMiddleware, init_default_shared
from common.defaults import DEFAULT_DOCUMENTS
import uuid


@asynccontextmanager
async def startup(app: FastAPI):
    default_shared = {
        "texts": list(DEFAULT_DOCUMENTS),
        "embeddings": None,
        "query": None,
        "query_embedding": None,
        "retrieved_document": None,
        "generated_answer": None
    }
    offline_flow.run(default_shared)
    init_default_shared(default_shared)
    yield


app = FastAPI(lifespan=startup)

app.add_middleware(SessionMiddleware)


@app.get("/")
async def homepage():
    return {"message": "Hello, World!"}


class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    shared = request.state.shared
    online_flow.run(shared)
    return ChatResponse(
        answer=shared.get("generated_answer", ""),
        session_id=request.state.session_id
    )


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

    shared = session_manager.get_shared(session_id)
    return UploadResponse(
        session_id=session_id,
        document_count=len(req.texts),
        chunk_count=len(shared.get("texts", []))
    )


if __name__ == "__main__":
    uvicorn.run("start_server:app", host="127.0.0.1", port=23333)
