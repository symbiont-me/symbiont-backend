from ..mongodb import studies_collection
from ..models import StudyResource
from .. import logger
from bson.objectid import ObjectId
from pymongo import ReturnDocument


class PlainTextRepo:
    def __init__(self, resource: StudyResource, user_id: str = "", study_id: ObjectId = None):
        self.user_id = user_id
        self.study_id = study_id
        self.resource = resource

    def add_study_resource_to_db(self):
        logger.debug("Adding to Mongo")
        # get study by id
        # add resource to study
        resource_to_add = {"resource_name": "resource_value"}  # replace with your resource to add

        result = studies_collection.find_one_and_update(
            {"_id": self.study_id},  # query to find the study by id
            {"$push": {"resources": resource}},  # update operation to add the resource to resources array
            return_document=ReturnDocument.AFTER,  # return the updated document
        )
        logger.info(result)
