from fastapi import APIRouter, Request
from pydantic import BaseModel
from firebase_admin import firestore

router = APIRouter()


class User(BaseModel):
    user_uid: str


default_llm_settings = {
    "llm_name": "gpt-3.5-turbo",
    "api_key": "",
    "temperature": 0.75,
    "max_tokens": 1500,
}


@router.post("/create-user-in-db")
async def create_user_in_db(user: User, request: Request):
    try:
        user_uid = request.state.verified_user["user_id"]
        db = firestore.client()
        doc_ref = db.collection("users").document(user_uid)
        # TODO set_default_llm_settings()
        if not doc_ref:
            return {"error": "User not found", "status_code": 404}

        new_user = User(user_uid=doc_ref.id)
        doc_ref.set(
            {
                "settings": {
                    "llm_name": default_llm_settings["llm_name"],
                    "api_key": default_llm_settings["api_key"],
                }
            }
        )
        doc_ref.set(new_user.model_dump())
        return {"message": "User created in db"}
    except Exception as e:
        return {"error": str(e), "status_code": 500}


@router.post("/get-user-by-uid")
async def get_user_by_uid(user_uid: str):
    try:
        db = firestore.client()
        doc_ref = db.collection("users").document(user_uid)
        user = doc_ref.get()
        return {"user": user.to_dict()}
    except Exception as e:
        return {"error": str(e), "status_code": 500}
