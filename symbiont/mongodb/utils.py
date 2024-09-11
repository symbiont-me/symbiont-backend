from fastapi import HTTPException
from symbiont.mongodb import studies_collection, users_collection
from pymongo.collection import Collection
from .. import logger


async def user_exists(user_id: str):
    user = users_collection.find_one({"_id": (user_id)})
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def check_user_authorization(studyId: str, user_uid: str, studies: Collection):
    try:
        studies = studies_collection.find_one({"_id": studyId})

        if studies is None:
            logger.error("Study not found")
            raise HTTPException(detail="Study Not Found", status_code=404)

        # Check if the user making the request is authorized
        if studies["userId"] != user_uid:
            logger.critical("User is Not authorized to delete study")
            raise HTTPException(
                detail="User Not Authorized to Delete Study",
                status_code=403,
            )
    except Exception as e:
        logger.error("An error occurred while checking user authorization")
        raise HTTPException(detail=str(e), status_code=500)
