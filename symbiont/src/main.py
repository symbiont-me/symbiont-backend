from fastapi import FastAPI
from .middleware.UserAuthVerify import AuthTokenMiddleware

from .routers import study as user_studies_router
from .routers import text as text_router
from .routers import chat as chat_router
from .routers import resource as resource_handling_router
from .routers import summary as summary_router
from .routers import llm_settings as llm_settings_router
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import re


class CORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = Response("Internal server error", status_code=500)
        pattern = re.compile(r"https://symbiont*thelonehegelian.vercel.app")
        origin = request.headers.get("origin")

        if origin and pattern.match(origin) or origin == "http://localhost:3000":
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = (
                "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range"
            )
            response.headers["Access-Control-Expose-Headers"] = "*"
            response.headers["Access-Control-Allow-Credentials"] = "true"

        return response


app = FastAPI()


# Add the AuthTokenMiddleware
app.add_middleware(AuthTokenMiddleware)


app.include_router(user_studies_router.router)
app.include_router(text_router.router)
app.include_router(chat_router.router)
app.include_router(resource_handling_router.router)
app.include_router(summary_router.router)
app.include_router(llm_settings_router.router)


# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "https://symbiont.vercel.app/",
#     "https://symbiont-git-main-thelonehegelian.vercel.app",
#     "https://symbiont-bfhadjedi-thelonehegelian.vercel.app",
#     "https://symbiont-6n13jr6cf-thelonehegelian.vercel.app",
# ]


app.add_middleware(CORSMiddleware)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# ~~~~~~~~~~~~~~~~~~~~~~~

# TODO for the library
# @app.post("get-user-resources")
