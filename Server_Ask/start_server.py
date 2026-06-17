"""
FastAPI 服务入口，启动时加载离线索引，提供 /chat 接口。
"""
from fastapi import FastAPI, HTTPException
import uvicorn
from contextlib import asynccontextmanager
from pydantic import BaseModel

from run_rag import offline_get_shared_store, online_get_shared_store
from rag_logic.flow import offline_flow, online_flow

@asynccontextmanager
async def startup(app: FastAPI):
    # 启动时构建离线索引（只执行一次）
    offline_shared_init = offline_get_shared_store()
    offline_flow.run(offline_shared_init)
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
    shared_to_online_flow = online_get_shared_store(req.query)
    online_flow.run(shared_to_online_flow)

    answer = shared_to_online_flow.get("generated_answer", "")
    if not answer:
        raise HTTPException(status_code=500, detail="Failed to generate answer")
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    uvicorn.run("start_server:app", host="127.0.0.1", port=23333)
