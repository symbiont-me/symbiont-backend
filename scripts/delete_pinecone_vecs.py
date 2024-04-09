# TODO update to get information from .env file
from pinecone import Pinecone
import os

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("PINECONE_API_KEY"):
    raise Exception("PINECONE_API_KEY is not set")

api_key = os.getenv("PINECONE_API_KEY")
index_name = "symbiont-dev"
region = "us-west-2"

pc = Pinecone(api_key=api_key, region=region)
index = pc.Index("symbiont-dev")
namespaces = []
if index:
    res = index.describe_index_stats()
    namespaces = list(res.get("namespaces", {}).keys())


for ns in namespaces:
    try:
        if index:
            index.delete(
                delete_all=True,
                namespace=ns,
            )
            print(f"Deleted namespace {ns}")
    except Exception as e:
        print(f"Error deleting namespace {ns}: {str(e)}")
        continue
