from .. import logger
import os
from gridfs import GridFS, GridFSBucket
import pymongo
from pymongo.server_api import ServerApi
from pymongo import MongoClient
import time


def init_db_collections(db):
    studies_collection = db["studies"]
    users_collection = db["users"]
    resources_collection = db["resources"]  # NOTE we'll use these after restructuring the database
    vectors_collection = db["vectors"]
    return studies_collection, users_collection


def init_mongo_db():
    mongo_uri = os.getenv("MONGO_URI")
    mongo_port = os.getenv("MONGO_PORT", 27017)
    mongo_db_name = os.getenv("MONGO_DB_NAME")

    if not all([mongo_uri, mongo_port, mongo_db_name]):
        raise Exception("MONGO_URI, MONGO_PORT, MONGO_DB_NAME environment variables are not set")

    client = None
    db = None
    try:
        logger.debug(f"Connecting to MongoDB at {mongo_uri}")
        if os.getenv("FASTAPI_ENV") == "development" and mongo_uri == "localhost":
            client = MongoClient(mongo_uri, int(mongo_port), serverSelectionTimeoutMS=5000)
        elif os.getenv("FASTAPI_ENV") == "development":
            client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        elif os.getenv("FASTAPI_ENV") == "production":
            client = MongoClient(mongo_uri, server_api=ServerApi("1"))
        # client.server_info()  # force connection on a request as the

        if client is None:
            raise pymongo.errors.ServerSelectionTimeoutError
        logger.debug("Pinging client ...")
        start_time = time.time()
        client.admin.command("ping")
        elapsed_time = time.time() - start_time
        logger.info(f"Pinged in {elapsed_time}! Connection to MongoDB was successful")
    except pymongo.errors.ServerSelectionTimeoutError as err:
        logger.error(f"Connection to MongoDB failed: {err}")

    if client is None:
        raise Exception("Connection to MongoDB failed")
    db = client[mongo_db_name]
    studies_collection, users_collection = init_db_collections(db)
    grid_fs = GridFS(db)
    grid_fs_bucket = GridFSBucket(db)

    return client, db, studies_collection, users_collection, grid_fs, grid_fs_bucket


client, db, studies_collection, users_collection, grid_fs, grid_fs_bucket = init_mongo_db()
