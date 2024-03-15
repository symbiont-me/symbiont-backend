from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .middleware.UserAuthVerify import AuthTokenMiddleware

from .routers import study as user_studies_router
from .routers import text as text_router
from .routers import chat as chat_router
from .routers import resource as resource_handling_router
from .routers import summary as summary_router
from .routers import llm_settings as llm_settings_router

app = FastAPI()


# Add the AuthTokenMiddleware
app.add_middleware(AuthTokenMiddleware)


app.include_router(user_studies_router.router)
app.include_router(text_router.router)
app.include_router(chat_router.router)
app.include_router(resource_handling_router.router)
app.include_router(summary_router.router)
app.include_router(llm_settings_router.router)


origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


# ~~~~~~~~~~~~~~~~~~~~~~~

# TODO for the library
# @app.post("get-user-resources")
