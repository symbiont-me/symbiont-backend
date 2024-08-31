from typing import List
from pymilvus import MilvusClient
from symbiont.models import DocumentPage
from pydantic import BaseModel
from symbiont.vector_dbs.models import VectorSearchResult
from ... import logger
from ..base_vector_repo_abc import BaseVectorRepository
from .. import vector_store_settings
from .. import embeddings_model
import uuid


class MilvusInsertResponse(BaseModel):
    insert_count: int
    ids: List


class Entity(BaseModel):
    text: str
    subject: str


class SearchResult(BaseModel):
    id: int
    distance: float
    entity: Entity


class MilvusSearchResponse(BaseModel):
    results: List[SearchResult]


class MilvusRepository(BaseVectorRepository):
    def __init__(self):
        self.dimension = vector_store_settings.configs.vector_store_dimension
        # TODO use localhost and cloud
        self.client = MilvusClient("milvus_demo.db")
        logger.info("Connected to Milvus")

    def __create_collection(self, collection_name: str, dimension: int) -> None:
        self.client.create_collection(collection_name, dimension)

    def __create_data_for_insertion(self, docs: List[DocumentPage], embeddings: List[List[float]]) -> List[dict]:
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": embeddings[i],
                "text": docs[i].page_content,
            }
            for i in range(len(docs))
        ]
        return points

    def __transform_search_results(self, search_results) -> List[VectorSearchResult]:
        return [VectorSearchResult(id=str(result.id), score=result.distance) for result in search_results]

    def __embed_single_document(self, doc: DocumentPage) -> List[float]:
        if embeddings_model is None:
            raise ValueError("Embeddings model not set")
        return embeddings_model.embed_query(doc.page_content)

    def upsert_vectors(self, namespace: str, docs: List[DocumentPage]) -> List[str]:
        vectors = [self.__embed_single_document(doc) for doc in docs]
        if not self.client.has_collection(collection_name=namespace):
            self.__create_collection(collection_name=namespace, dimension=int(self.dimension))
        points = self.__create_data_for_insertion(docs, vectors)
        insert_response: MilvusInsertResponse = self.client.insert(collection_name=namespace, data=points)  # type: ignore
        return insert_response.ids

    def search_vectors(self, namespace: str, query: str, limit: int) -> List[VectorSearchResult]:
        if embeddings_model is None:
            raise ValueError("Embeddings model not initialized")
        vectorised_query = embeddings_model.embed_query(query)
        search_response: MilvusSearchResponse = self.client.search(
            collection_name=namespace, query_vector=vectorised_query, limit=limit
        )  # type: ignore
        transformed_results = self.__transform_search_results(search_response.results)
        logger.info(f"Found {len(transformed_results)} results")
        return transformed_results

    def delete_vectors(self, namespace: str) -> None:
        self.client.drop_collection(collection_name=namespace)
        logger.info(f"Deleted: Vectors for {namespace}")
