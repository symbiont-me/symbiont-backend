from fastapi import APIRouter, Request
from firebase_admin import firestore
from ..mongodb import users_collection
from ..models import UserCollection
from .. import logger

router = APIRouter()


# Note: since we are using a single Firebase Google sign_in method for signing up and sign in
# we will use this single route
# Later we can add a separate route for sign up if we use other sign up methods
@router.post("/login")
async def login_user(request: Request):
    try:
        user_uid = request.state.verified_user["user_id"]
        # if user already exists return
        user = users_collection.find_one({"_id": user_uid})
        if user:
            logger.info(f"User already exists in db with id {user_uid}")
            return {"message": "User already exists in db"}

        new_user = UserCollection(studies=[], settings={})

        results = users_collection.insert_one({"_id": user_uid, **new_user.model_dump()})
        logger.info(f"User created in db with id {results.inserted_id}")

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
