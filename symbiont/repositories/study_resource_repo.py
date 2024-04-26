from ..models import StudyResource
from .. import logger
from pymongo import ReturnDocument
from ..mongodb import studies_collection


class StudyResourceRepo:
    def __init__(self, resource: StudyResource, user_id: str = "", study_id: str = ""):
        self.user_id = user_id
        self.study_id = study_id
        self.resource = resource

    def add_study_resource_to_db(self):
        logger.debug("Adding to Mongo")
        result = studies_collection.find_one_and_update(
            {"_id": self.study_id},
            {"$push": {"resources": self.resource.model_dump()}},
            return_document=ReturnDocument.AFTER,
        )

        logger.info(result)
