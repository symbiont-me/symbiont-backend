from fastapi import APIRouter
from fastapi import HTTPException
from fastapi import Request
from ..utils.db_utils import StudyService

router = APIRouter()


def get_summaries_from_db(studyId, user_uid):
    study_service = StudyService(user_uid, studyId)
    study_dict = study_service.get_document_dict()
    if study_dict is None:
        raise HTTPException(status_code=404, detail="Study not found")
    resources = study_dict.get("resources", [])
    summaries = [
        {
            "name": resource.get("name"),
            "identifier": resource.get("identifier"),
            "url": resource.get("url"),
            "summary": resource.get("summary"),
        }
        for resource in resources
    ]
    return {"summaries": summaries}


# TODO get summaries route
@router.get("/get-summaries")
async def get_summaries(studyId: str, request: Request):
    user_uid = request.state.verified_user["user_id"]
    summaries = get_summaries_from_db(studyId, user_uid)
    return summaries
