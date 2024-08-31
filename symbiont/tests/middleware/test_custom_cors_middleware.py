from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from symbiont.middleware.CustomCorsMiddleware import CORSMiddleware


app = FastAPI()
app.add_middleware(CORSMiddleware)


@app.get("/")
async def root():
    return {"message": "Hello, World!"}


client = TestClient(app)


def test_cors_middleware_with_allowed_origin():
    response = client.get("/", headers={"origin": "https://symbiontthelonehegelian.vercel.app"})
    assert response.status_code == 200
    assert response.headers["Access-Control-Allow-Origin"] == "https://symbiontthelonehegelian.vercel.app"
    assert response.headers["Access-Control-Allow-Methods"] == "GET, POST, OPTIONS"
    assert response.headers["Access-Control-Allow-Headers"] == (
        "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range"
    )
    assert response.headers["Access-Control-Expose-Headers"] == "*"
    assert response.headers["Access-Control-Allow-Credentials"] == "true"


def test_cors_middleware_with_disallowed_origin():
    response = client.get("/", headers={"origin": "https://unauthorized.origin.com"})
    assert response.status_code == 500
    assert response.text == "Internal server error"


def test_cors_middleware_without_origin():
    response = client.get("/")
    assert response.status_code == 500
    assert response.text == "Internal server error"


if __name__ == "__main__":
    pytest.main()
