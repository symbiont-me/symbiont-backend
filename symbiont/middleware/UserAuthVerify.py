from fastapi import Request
from fastapi.responses import JSONResponse
from typing import Callable, Optional
from starlette.middleware.base import BaseHTTPMiddleware
from supertokens_python.recipe.session.framework.fastapi import verify_session


# class AuthTokenMiddleware(BaseHTTPMiddleware):
#     """
#     Middleware for verifying and extracting user authentication token from the Authorization header.

#     Attributes:
#         ROUTES_TO_EXCLUDE (list): List of routes to exclude from authentication.
#     """

#     def __init__(self, app):
#         """
#         Initializes the middleware.

#         Args:
#             app: The FastAPI application instance.
#         """
#         super().__init__(app)
#         self.ROUTES_TO_EXCLUDE = ["/status", "/docs", "/redoc", "/openapi.json"]  # We've removed "/" for now

#     async def dispatch(self, request: Request, call_next: Callable) -> JSONResponse:
#         """
#         Dispatch method that verifies the Authorization header and extracts the user authentication token.

#         If the token is valid, it sets the decoded token in the request state.

#         Args:
#             request: The incoming request.
#             call_next: The next middleware or route to call.

#         Returns:
#             A JSON response with the result of the authentication.
#         """

#         # Skip auth middleware if path is one of the excluded routes
#         if request.url.path in self.ROUTES_TO_EXCLUDE:
#             return await call_next(request)

#         authorization: Optional[str] = request.headers.get("Authorization")

#         if not authorization:
#             return JSONResponse(status_code=401, content={"details": "Authorization header missing"})

#         try:
#             id_token = authorization.split("Bearer ")[1]
#         except IndexError:
#             return JSONResponse(status_code=401, content={"details": "Invalid Authorization header"})

#         try:
#             decoded_token = auth.verify_id_token(id_token)
#             request.state.verified_user = decoded_token
#         except Exception as e:
#             return JSONResponse(status_code=401, content={"details": str(e)})

#         response = await call_next(request)
#         return response
