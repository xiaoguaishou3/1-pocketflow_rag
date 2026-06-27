"""
会话中间件：处理 session 逻辑，预构建 shared 存入 request.state。
"""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from sessions.session import session_manager


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/chat" and request.method == "POST":
            try:
                body = await request.json()
                query = body.get("query", "")
                session_id = body.get("session_id") or str(uuid.uuid4())

                chunks = session_manager.get_chunks_for_session(session_id)
                shared = session_manager.build_shared(query, chunks)

                request.state.shared = shared
                request.state.session_id = session_id
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Request parsing failed: {str(e)}"}
                )

        response = await call_next(request)
        return response
