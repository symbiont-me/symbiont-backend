from fastapi import APIRouter
from ..models import TextUpdateRequest
from firebase_admin import firestore


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       STUDY TEXT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

router = APIRouter()


@router.post("/update-text")
async def update_text(text: TextUpdateRequest):
    # TODO verify user has access to study
    db = firestore.client()
    study_ref = db.collection("studies_").document(text.studyId)
    study = study_ref.get()
    if study.exists:
        study_ref.update({"text": text.text})
        return {"message": "Text updated successfully"}
    else:
        print("No such document!")
        return {"message": "No such document!"}
