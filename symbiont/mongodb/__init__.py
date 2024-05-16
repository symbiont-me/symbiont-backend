from pymongo import MongoClient
from .. import logger
import os
from pydantic import BaseModel

from gridfs import GridFS, GridFSBucket
import pymongo


def init_db_collections(db):
    studies_collection = db["studies"]
    users_collection = db["users"]
    resources_collection = db["resources"]
    vectors_collection = db["vectors"]
    return studies_collection, users_collection


if os.getenv("FASTAPI_ENV") == "development":
    client = None
    try:
        client = pymongo.MongoClient("localhost", 27017, serverSelectionTimeoutMS=5000)  # add timeout for connection
        client.server_info()  # force connection on a request as the
        # connect=True parameter of MongoClient seems
        # to be useless here
        print("Connection to MongoDB was successful")
    except pymongo.errors.ServerSelectionTimeoutError as err:
        logger.error("Connection to MongoDB failed")
    # TODO add to env file
    if client:
        db = client["symbiont-dev"]
        print(db)
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
