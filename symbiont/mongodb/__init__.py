from pymongo import MongoClient
from .. import logger

logger.debug("Connecting to MongoDB")

client = MongoClient("localhost", 27017)
logger.info("Connected to Mongodb")

db = client["symbiont-dev"]
user_collection = db["users"]

logger.debug(user_collection.find_one())


# TODO add condition for production
