import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from httpx import AsyncClient
from symbiont.middleware.UserAuthVerify import SuperTokensAuthMiddleware


# Mock verify_session function
async def mock_verify_session():
    class MockSession:
        pass

    return MockSession()


@pytest.fixture
def app():
    app = FastAPI()

    @app.get("/status")
    async def status():
        return {"message": "OK"}

    @app.get("/protected")
    async def protected(request: Request):
        return {"message": "Protected", "session": request.state.session}

    app.add_middleware(SuperTokensAuthMiddleware)
    return app


@pytest.mark.asyncio
async def test_initialization(app):
    assert app


@pytest.mark.asyncio
async def test_route_exclusion(app):
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/status")
    assert response.status_code == 200
    assert response.json() == {"message": "OK"}


# TODO needs to be fixed
# @pytest.mark.asyncio
# async def test_valid_token(app, monkeypatch):
#     async def mock_verify_session_wrapper():
#         return await mock_verify_session()

#     monkeypatch.setattr("symbiont.middleware.UserAuthVerify.verify_session", lambda: mock_verify_session_wrapper)

#     async with AsyncClient(app=app, base_url="http://test") as ac:
#         response = await ac.get("/protected")
#     assert response.status_code == 200
#     assert response.json() == {"message": "Protected", "session": {}}


@pytest.mark.asyncio
async def test_invalid_token(app, monkeypatch):
    async def mock_verify_session_fail():
        raise Exception("Invalid token")

    monkeypatch.setattr("symbiont.middleware.UserAuthVerify.verify_session", lambda: mock_verify_session_fail)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/protected")
    assert response.status_code == 401
    assert response.json() == {"message": "Unauthorized"}


@pytest.mark.asyncio
async def test_error_handling(app, monkeypatch):
    async def mock_verify_session_error():
        raise Exception("Some error")

    monkeypatch.setattr("symbiont.middleware.UserAuthVerify.verify_session", lambda: mock_verify_session_error)

    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/protected")
    assert response.status_code == 401
    assert response.json() == {"message": "Unauthorized"}
