from .. import logger
import os
from gridfs import GridFS, GridFSBucket
import pymongo
from pymongo.server_api import ServerApi


def init_db_collections(db):
    studies_collection = db["studies"]
    users_collection = db["users"]
    resources_collection = db["resources"]
    vectors_collection = db["vectors"]
    return studies_collection, users_collection


mongo_uri = os.getenv("MONGO_URI")
mongo_port = os.getenv("MONGO_PORT")
mongo_db_name = os.getenv("MONGO_DB_NAME")

client = None

try:
    if os.getenv("FASTAPI_ENV") == "development":
        client = pymongo.MongoClient(
            mongo_uri, int(mongo_port), serverSelectionTimeoutMS=5000
        )  # add timeout for connection
    elif os.getenv("FASTAPI_ENV") == "production":
        client = pymongo.MongoClient(mongo_uri, server_api=ServerApi("1"))
    # client.server_info()  # force connection on a request as the

    if client is None:
        raise pymongo.errors.ServerSelectionTimeoutError
    client.admin.command("ping")
    logger.info("Pinged! Connection to MongoDB was successful")
except pymongo.errors.ServerSelectionTimeoutError as err:
    logger.error("Connection to MongoDB failed")


if client:
    db = client[mongo_db_name]
    studies_collection, users_collection = init_db_collections(db)
    grid_fs = GridFS(db)
    grid_fs_bucket = GridFSBucket(db)
