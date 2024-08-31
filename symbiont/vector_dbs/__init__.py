"""
TODO: Needs to be updated to make methods async where appropriate
TODO: The whole file needs to be refactored.
TODO: Param and Return types need to be specified
TODO: Clear documentation needs to be added, with regards to purpose of the classes and the file
NOTE: The main interfaces should be maintained, because it allows for easy addition of new vector databases
"""

import os
from symbiont.vector_dbs.embeddings_service import EmbeddingsService
from symbiont.vector_dbs.reranker_service import RerankerService


from .models import ConfigsModel


class VectorStoreSettings:
    def __init__(self):
        self.configs = ConfigsModel(
            vector_store=os.getenv("VECTOR_STORE", ""),
            vector_store_url=os.getenv("VECTOR_STORE_URL", ""),
            vector_store_port=os.getenv("VECTOR_STORE_PORT", ""),
            vector_store_dimension=os.getenv("VECTOR_STORE_DIMENSION", "1536"),
            vector_store_distance=os.getenv("VECTOR_STORE_DISTANCE", ""),
            vector_store_token=os.getenv("VECTOR_STORE_TOKEN"),
            embeddings_model=os.getenv("EMBEDDINGS_MODEL", "bge-base-en"),
        )


vector_store_settings = VectorStoreSettings()


embeddings_service = EmbeddingsService(vector_store_settings.configs)
embeddings_model, text_splitter, nltk_text_splitter = (
    embeddings_service.embeddings_model,
    embeddings_service.text_splitter,
    embeddings_service.nltk_text_splitter,
)

reranker_service = RerankerService("cohere")
reranker = reranker_service.reranker
