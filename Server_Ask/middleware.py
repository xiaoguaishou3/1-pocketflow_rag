"""
会话中间件：处理 session 逻辑，获取 shared 存入 request.state。
"""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from sessions.session import session_manager


_default_shared = None


def init_default_shared(shared: dict):
    global _default_shared
    _default_shared = shared


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/chat" and request.method == "POST":
            try:
                body = await request.json()
                query = body.get("query", "")
                session_id = body.get("session_id") or str(uuid.uuid4())

                shared = session_manager.get_shared(session_id)
                if shared is None:
                    shared = dict(_default_shared)

                shared["query"] = query
                request.state.shared = shared
                request.state.session_id = session_id
            except Exception as e:
                return JSONResponse(
                    status_code=400,
                    content={"detail": f"Request parsing failed: {str(e)}"}
                )

        response = await call_next(request)
        return response
