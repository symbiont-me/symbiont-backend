from pymongo import MongoClient
from .. import logger
import os
from pydantic import BaseModel

from gridfs import GridFS, GridFSBucket


def init_db_collections(db):
    studies_collection = db["studies"]
    users_collection = db["users"]
    resources_collection = db["resources"]
    vectors_collection = db["vectors"]
    return studies_collection, users_collection


if os.getenv("FASTAPI_ENV") == "development":
    client = MongoClient("localhost", 27017)
    logger.info("Connected to Mongodb")
    # TODO add to env file
    db = client["symbiont-dev"]
    studies_collection, users_collection = init_db_collections(db)
    grid_fs = GridFS(db)
    grid_fs_bucket = GridFSBucket(db)


if os.getenv("FASTAPI_ENV") == "production":
    client = MongoClient(os.getenv("MONGO_URI"), 27017)
    logger.info("Connected to Mongodb")
    # TODO add to env file
    db = client[os.getenv("MONGO_DB_NAME")]
    studies_collection, users_collection = init_db_collections(db)
    grid_fs = GridFS(db)
    grid_fs_bucket = GridFSBucket(db)
