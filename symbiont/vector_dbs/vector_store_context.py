from . import vector_store_settings
from .vector_store_repos.qdrant_repo import QdrantRepository

vector_store_repos = {
    "qdrant": QdrantRepository,
}


class VectorStoreContext:
    def __init__(self):
        self.vector_store = vector_store_settings.configs.vector_store
        if vector_store_settings.configs.vector_store is None:
            raise ValueError("Set the Vector Store name")
        if self.vector_store not in vector_store_repos:
            raise ValueError("Vector store not supported")
        # NOTE this instiates the vector store repo using the object
        self.vector_store_repo = vector_store_repos[vector_store_settings.configs.vector_store]()
