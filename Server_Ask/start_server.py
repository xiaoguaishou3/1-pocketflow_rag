from fastapi import FastAPI
import uvicorn

from run_rag import offline_get_shared_store, online_get_shared_store
from rag_logic.flow import offline_flow, online_flow

app = FastAPI()

@app.on_event("startup")
async def startup():
    shared = offline_get_shared_store()
    offline_flow.run(shared)

@app.get("/")
async def homepage():
    return {"message": "Hello, World!"}

@app.post("/chat")
async def chat(query: str):
    print("=======\n", query)
    shared = online_get_shared_store(query)
    online_flow.run(shared)

    return "ok"
    

if __name__ == "__main__":
    uvicorn.run("start_server:app", host="127.0.0.1", port=23333)