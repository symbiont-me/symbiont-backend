from fastapi import FastAPI, Depends

# from .middleware.UserAuthVerify import AuthTokenMiddleware
from fastapi.middleware.cors import CORSMiddleware
from .routers import study as user_studies_router
from .routers import text as text_router
from .routers import chat as chat_router
from .routers import resource as resource_handling_router
from .routers import summary as summary_router
from .routers import llm_settings as llm_settings_router
from .routers import tests as tests_router
from .routers import user as user_router
from . import ENVIRONMENT, VERSION
from supertokens_python.recipe import session
from supertokens_python.recipe.session.framework.fastapi import verify_session
from supertokens_python import get_all_cors_headers
from supertokens_python.framework.fastapi import get_middleware
from symbiont.supertokens import init_supertokens
from starlette.middleware.base import BaseHTTPMiddleware
from langchain import PromptTemplate
from pydantic import BaseModel

# from symbiont.supertokens import init_supertokens

# NOTE: Make sure to update version when there is a major change in code!

WELCOME_MESSAGE = (
    "Welcome to Symbiont's API! Check out the docs at /docs, "
    "the frontend at https://symbiont.vercel.app "
    "and the source code on https://github.com/symbiont-me"
)


app = FastAPI()


class SessionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        session = await verify_session()(request)
        request.state.session = session
        return await call_next(request)


init_supertokens()

app.add_middleware(SessionMiddleware)
app.add_middleware(get_middleware())

# app.add_middleware(AuthTokenMiddleware)

app.include_router(user_studies_router.router)
app.include_router(text_router.router)
app.include_router(chat_router.router)
app.include_router(resource_handling_router.router)
app.include_router(summary_router.router)
app.include_router(llm_settings_router.router)
app.include_router(user_router.router)
app.include_router(tests_router.router)  # This is for testing purposes only
# Don't add a trailing forward slash to a url in origins! Will cause CORS issues
origins = [
    "http://localhost:3003",
    "http://localhost:3000",
    "https://symbiont.vercel.app",
    "https://staging-symbiont.vercel.app",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "PUT", "POST", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Content-Type", "Accept-Language"] + get_all_cors_headers(),
)


@app.get("/status")
async def status_check():
    status_response = {
        "status": "up",
        "version": VERSION,
        "environemnt": ENVIRONMENT,
    }
    return status_response


@app.get("/session-details/")
async def session_details(
    session: session.SessionContainer = Depends(verify_session()),
):
    session_data = {
        "user_id": session.get_user_id(),
        "session_handle": session.get_handle(),
        "access_token_payload": session.get_access_token_payload(),
    }
    print(session_data)
    return {"session_data": session_data}


@app.get("/")
async def read_root():
    return {"message": WELCOME_MESSAGE}
