from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from typing import Callable
from firebase_admin import auth
from starlette.middleware.base import BaseHTTPMiddleware


class AuthTokenMiddleware(BaseHTTPMiddleware):
    """
    Middleware for verifying and extracting user authentication token from the Authorization header.
    """

    def __init__(self, app):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable):
        """
        Dispatch method that verifies the Authorization header and extracts the user authentication token.
        If the token is valid, it sets the decoded token in the request state.
        """

        authorization: str = request.headers.get("Authorization", "")
        if authorization is None:
            return JSONResponse(
                status_code=401, content={"detail": "Authorization header missing"}
            )

        try:
            id_token = authorization.split("Bearer ")[1]
            decoded_token = auth.verify_id_token(id_token)
            request.state.verified_user = decoded_token
        except Exception as e:
            return JSONResponse(status_code=401, content={"detail": str(e)})

        response = await call_next(request)
        return response
