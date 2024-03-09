from fastapi import APIRouter
from pydantic import BaseModel
from firebase_admin import firestore

router = APIRouter()


class User(BaseModel):
    user_uid: str


@router.post("/create-user-in-db")
async def create_user_in_db(user: User):
    try:
        db = firestore.client()
        doc_ref = db.collection("users").document()
        new_user = User(user_uid=doc_ref.id)
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
