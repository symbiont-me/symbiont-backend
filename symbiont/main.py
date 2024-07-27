from fastapi import FastAPI
from .middleware.UserAuthVerify import AuthTokenMiddleware
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

app = FastAPI()


# Add the AuthTokenMiddleware
app.add_middleware(AuthTokenMiddleware)


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
    "http://localhost",
    "http://localhost:3000",
    "https://symbiont.vercel.app",
    "https://staging-symbiont.vercel.app",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "X-Auth-Token",
        "X-User-Identifier",
    ],
)


@app.get("/status")
async def status_check():
    status_response = {
        "status": "up", 
        "version": VERSION,
        "environemnt": ENVIRONMENT,
    }
    return status_response


@app.get("/")
async def read_root():
    return {"message": "Welcome to symbiont api! Check out the docs at /docs, frontend at https://symbiont.vercel.app and the source code on https://github.com/symbiont-me"}
