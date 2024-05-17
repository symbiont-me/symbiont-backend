from logging import raiseExceptions
from .. import logger
import os
from gridfs import GridFS, GridFSBucket
import pymongo
from pymongo.server_api import ServerApi
from pymongo import MongoClient


def init_db_collections(db):
    studies_collection = db["studies"]
    users_collection = db["users"]
    resources_collection = db["resources"]  # NOTE we'll use these after restructuring the database
    vectors_collection = db["vectors"]
    return studies_collection, users_collection


def init_mongo_db():
    mongo_uri = os.getenv("MONGO_URI")
    mongo_port = os.getenv("MONGO_PORT")
    mongo_db_name = os.getenv("MONGO_DB_NAME")

    if not mongo_uri or not mongo_port or not mongo_db_name:
        logger.error("MONGO Settings are not set in the environment variable")
        raise ValueError("MONGO Settings are not set in the environment variable")

    client = None
    db = None

    try:
        if os.getenv("FASTAPI_ENV") == "development":
            client = MongoClient(
                mongo_uri, int(mongo_port), serverSelectionTimeoutMS=5000
            )  # add timeout for connection
        elif os.getenv("FASTAPI_ENV") == "production":
            client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        # client.server_info()  # force connection on a request as the

        if client is None:
            raise pymongo.errors.ServerSelectionTimeoutError
        client.admin.command("ping")
        logger.info("Pinged! Connection to MongoDB was successful")
    except pymongo.errors.ServerSelectionTimeoutError as err:
        logger.error("Connection to MongoDB failed")

    if client is None:
        raise pymongo.errors.ServerSelectionTimeoutError

    db = client[mongo_db_name]
    studies_collection, users_collection = init_db_collections(db)
    grid_fs = GridFS(db)
    grid_fs_bucket = GridFSBucket(db)

    return client, db, studies_collection, users_collection, grid_fs, grid_fs_bucket


client, db, studies_collection, users_collection, grid_fs, grid_fs_bucket = init_mongo_db()
