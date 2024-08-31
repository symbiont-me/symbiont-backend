from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import re


# TODO this needs to be updated, especially the pattern
class CORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        pattern = re.compile(r"https://symbiont*thelonehegelian.vercel.app")
        origin = request.headers.get("origin")

        if origin and pattern.match(origin):
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range"
            )
            response.headers["Access-Control-Expose-Headers"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"
        else:
            response = Response("Internal server error", status_code=500)

        return response
