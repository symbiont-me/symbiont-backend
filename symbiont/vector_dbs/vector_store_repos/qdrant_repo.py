from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.models import Batch
import uuid
from symbiont.models import DocumentPage
from ... import logger
from typing import List
from ..base_vector_repo_abc import BaseVectorRepository, VectorSearchResult
from .. import vector_store_settings
from .. import embeddings_model


class QdrantRepository(BaseVectorRepository):
    def __init__(self):
        self.dimension = vector_store_settings.vector_store_dimension
        self.distance = vector_store_settings.vector_store_distance
        self.client = QdrantClient(
            url=f"{vector_store_settings.vector_store_url}:{vector_store_settings.vector_store_port}",
            port=6333,
            api_key=vector_store_settings.vector_store_token,
        )

        logger.info(
            f"Connected to Qdrant at {vector_store_settings.vector_store_url}:{vector_store_settings.vector_store_port}"
        )

    def create_collection(self, collection_name: str, vector_size: int, distance: str) -> None:
        distance = Distance.DOT if distance.lower() == "dot" else Distance.COSINE

        is_collection = self.client.collection_exists(collection_name=collection_name)
        if is_collection is False:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.DOT),
            )

    def __create_points_for_upsert(self, docs: List[DocumentPage], embeddings: List[List[float]]) -> Batch:
        payload_list = [{"page_content": ""} for _ in docs]
        points = Batch(
            ids=[str(uuid.uuid4()) for _ in range(len(docs))],
            vectors=embeddings,
            payloads=payload_list,
        )
        return points

    def embed_single_document(self, doc: DocumentPage) -> List[float]:
        # TODO this should be abstracted as it won't work with non-langchain models
        if embeddings_model is None:
            raise ValueError("Embeddings model not set")
        return embeddings_model.embed_query(doc.page_content)

    def upsert_vectors(self, namespace: str, docs: List[DocumentPage]) -> List[str]:
        vectors = []
        for doc in docs:
            vec = self.embed_single_document(doc)
            vectors.append(vec)
        print(len(vectors))
        # vectors = embeddings_service.create_docs_embeddings(docs)
        #
        if self.client.collection_exists(collection_name=namespace) is False:
            self.create_collection(
                collection_name=namespace,
                vector_size=1536,  # TODO use vector_store_settings.vector_store_dimension
                distance=Distance.DOT,
            )
        #
        points = self.__create_points_for_upsert(docs, vectors)
        self.client.upsert(collection_name=namespace, points=points)
        # TODO check if there were no errors
        # @note points.ids is a type of List[ExtendedPointId]
        # we just want to keep this standard for other vector store implementations
        return [str(point_id) for point_id in points.ids]

    # TODO create a ScoredPoint type
    def __transform_search_results(self, search_results) -> List[VectorSearchResult]:
        return [VectorSearchResult(id=result.id, score=result.score) for result in search_results]

    def search_vectors(self, namespace: str, query: str, limit: int) -> List[VectorSearchResult]:
        if embeddings_model is None:
            raise ValueError("Embeddings model not initialized")
        vectorised_query = embeddings_model.embed_query(query)
        results = self.client.search(collection_name=namespace, query_vector=vectorised_query, limit=limit)
        transformed_results = self.__transform_search_results(results)
        logger.info(f"Found {len(transformed_results)} results")
        return transformed_results

    def delete_vectors(self, namespace: str) -> None:
        self.client.delete_collection(collection_name=namespace)
        logger.info(f"Deleted: Vectors for {namespace}")
