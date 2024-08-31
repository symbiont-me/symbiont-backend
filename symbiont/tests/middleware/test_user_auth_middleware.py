import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.testclient import TestClient
from symbiont.middleware.UserAuthVerify import AuthTokenMiddleware
from firebase_admin import auth


# Mock Firebase Admin auth.verify_id_token
def mock_verify_id_token(token):
    if token == "valid_token":
        return {"uid": "12345"}
    raise Exception("Invalid token")


# Apply the mock
auth.verify_id_token = mock_verify_id_token

app = FastAPI()


@app.get("/status")
async def status_route():
    return JSONResponse(content={"status": "ok"})


@app.get("/protected")
async def protected_route(request: Request):
    return JSONResponse(content={"message": "Protected route", "user": request.state.verified_user})


app.add_middleware(AuthTokenMiddleware)

client = TestClient(app)


def test_no_authorization_header():
    response = client.get("/protected")
    assert response.status_code == 401
    assert response.json() == {"details": "Authorization header missing"}


def test_invalid_authorization_header():
    response = client.get("/protected", headers={"Authorization": "InvalidHeader"})
    assert response.status_code == 401
    assert response.json() == {"details": "Invalid Authorization header"}


def test_invalid_token():
    response = client.get("/protected", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 401
    assert response.json() == {"details": "Invalid token"}


def test_valid_token():
    response = client.get("/protected", headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 200
    assert response.json() == {"message": "Protected route", "user": {"uid": "12345"}}


def test_excluded_route():
    response = client.get("/status")
    assert response.status_code == 200
