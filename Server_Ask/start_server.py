"""
FastAPI 服务入口，启动时加载离线索引，提供 /chat 接口。
"""
from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel

from run_rag import offline_get_shared_store, online_get_shared_store
from rag_logic.flow import offline_flow, online_flow

@asynccontextmanager
async def startup(app: FastAPI):
    shared = offline_get_shared_store()
    offline_flow.run(shared)
    yield

app = FastAPI(lifespan=startup)


@app.get("/")
async def homepage():
    return {"message": "Hello, World!"}


class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    print(f"=======\n{req}")
    shared = online_get_shared_store(req.query)
    online_flow.run(shared)

    return ChatResponse(answer="结果前面打印出来了")


if __name__ == "__main__":
    uvicorn.run("start_server:app", host="127.0.0.1", port=23333)
