from fastapi import APIRouter

router = APIRouter()


# TODO save resource content in the DB on upload?
def get_resource_content(identifier: str):
    # TODO search for resource in the DB using the identifier
    # return the text_content
    pass


# TODO this should be a background task onUpload and update the state on the frontend when ready
def make_summaries(text_content: str):
    # TODO get all resources text content
    # TODO make summaries one by one
    # TODO save summaries with identifier and name of the resource in DB
    pass


def get_summaries(studyId):
    # TODO return all the summaries as an array
    pass


# TODO get summaries route
