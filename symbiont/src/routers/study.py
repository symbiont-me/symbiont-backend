from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from ..models import CreateStudyRequest
from firebase_admin import firestore
from ..utils.db_utils import get_document_ref
from datetime import datetime
from ..models import Study, Chat

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       USER STUDIES
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


router = APIRouter()


@router.get("/get-user-studies")
async def get_user_studies(request: Request):
    user_uid = request.state.verified_user["user_id"]
    # TODO fix pattern for returning data, cf. other routers
    try:
        db = firestore.client()
        studies_ref = db.collection("studies_")
        query = studies_ref.where("userId", "==", user_uid)
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


@router.post("/create-study")
async def create_study(study: CreateStudyRequest, request: Request):
    user_uid = request.state.verified_user["user_id"]

    new_study = Study(
        name=study.name,
        description=study.description,
        userId=user_uid,
        image=study.image,
        createdAt=str(datetime.now()),
        resources=[],
        chat=Chat(),
    )
    try:
        db = firestore.client()
        doc_ref = db.collection("studies_").document()
        doc_ref.set(new_study.model_dump())

        return {"message": "Study created successfully", "study": study.model_dump()}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "message": "An error occurred while creating the study.",
                "details": str(e),
            },
        )


@router.delete("/delete-study")
async def delete_study(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    study_ref = get_document_ref("studies_", "userId", user_uid, studyId)
    if study_ref is None:
        raise HTTPException(status_code=404, detail="No such document!")
    study_ref.delete()
    return {"message": "Study deleted successfully", "status_code": 200}


@router.get("/get-study")
async def get_study(studyId: str, request: Request):
    # user_uid = request.state.verified_user["user_id"]
    # study_ref = get_document_ref("studies_", "userId", user_uid, studyId)
    # if study_ref is None:
    # raise HTTPException(status_code=404, detail="No such document!")
    # study = study_ref.get()
    # if study.exists:
    # return {"study": study.to_dict()}
    return {"study": "study"}
