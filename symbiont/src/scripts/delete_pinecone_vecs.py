import os
from pinecone import Pinecone

api_key = os.getenv("PINECONE_API_KEY")
index_name = os.getenv("PINECONE_INDEX_NAME")
region = os.getenv("PINECONE_REGION")

pc = Pinecone(api_key=api_key, region=region)
index = pc.Index("symbiont-me")
if index:
    index.delete(delete_all=True, namespace="20240219040159_1712.01210v1.pdf")
