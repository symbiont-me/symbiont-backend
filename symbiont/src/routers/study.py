from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse
from ..models import Study
from ..utils import verify_token
from firebase_admin import firestore

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       USER STUDIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


router = APIRouter()


@router.post("/get-user-studies")
async def get_user_studies(decoded_token: dict = Depends(verify_token)):
    try:
        userId = decoded_token["uid"]
        db = firestore.client()
        studies_ref = db.collection("studies_")
        query = studies_ref.where("userId", "==", userId)
        studies = query.stream()

        # Create a list of dictionaries, each containing the studyId and the study's data
        studies_data = [
            {"id": study.id, **(study.to_dict() or {})} for study in studies
        ]
        return {"studies": studies_data}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "An error occurred while fetching user studies.",
                "details": str(e),
            },
        )


@router.post("/create-study/")
async def create_study(study: Study):
    db = firestore.client()
    doc_ref = db.collection("studies_").document()
    doc_ref.set(study.model_dump())

    return {"message": "Study created successfully", "study": study.model_dump()}


@router.post("/get-study/")
async def get_study(studyId: str):
    db = firestore.client()
    # TODO verify user has access to study
    study_ref = db.collection("studies_").document(studyId)
    study = study_ref.get()
    if study.exists:
        print("Document data:", study.to_dict())
        return {"study": study.to_dict()}
    else:
        print("No such document!")
        return {"message": "No such document!", "study": {}}
