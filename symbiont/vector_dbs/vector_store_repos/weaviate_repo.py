from typing import List
import weaviate
from weaviate.embedded import EmbeddedOptions
from symbiont.vector_dbs.base_vector_repo_abc import BaseVectorRepository
from symbiont.vector_dbs.models import VectorSearchResult
from symbiont.models import DocumentPage
from .. import embeddings_model


class WeaviateRepository(BaseVectorRepository):
    # TODO refactor this like Qdrant repo
    def __init__(self):
        self.client = self.init_weaviate()

    def init_weaviate(self):
        client = weaviate.Client(
            embedded_options=EmbeddedOptions(
                persistence_data_path="symbiont/weaviate_data",
            )
        )
        print("is_ready", client.is_ready())
        return client

    def upsert_vectors(self, namespace: str, docs: List[DocumentPage]) -> List:
        if embeddings_model is None:
            raise ValueError("Embedding model is not set")
        # create a schema
        class_obj = {"class": namespace, "vectorizer": "none"}
        if self.client.schema.exists(namespace) is False:
            self.client.schema.create_class(class_obj)
        self.client.batch.configure(batch_size=len(docs), dynamic=True)

        results = []
        # upsert the vectors
        with self.client.batch as batch:
            for _, doc in enumerate(docs):
                properties = {"text": doc.page_content}
                vector = embeddings_model.embed_query(doc.page_content)
                results.append(batch.add_data_object(data_object=properties, vector=vector, class_name=namespace))

        return results

    def search_vectors(self, namespace: str, query: str, limit: int) -> List[VectorSearchResult]:
        if embeddings_model is None:
            raise ValueError("Embedding model is not set")
        print(f"Query: {query}")
        query_vector = embeddings_model.embed_query(query)
        print(len(query_vector))
        if self.client.schema.exists(namespace) is False:
            raise ValueError("Namespace does not exist")
        result = (
            self.client.query.get(namespace, ["text"])
            .with_near_vector({"vector": query_vector, "certainty": 0.7})
            .with_limit(2)
            .with_additional(["certainty", "distance"])
            .do()
        )
        vector_id = []
        for obj in result["data"]["Get"]["YourClassName"]:
            vector_id.append(obj["_additional"]["id"])

        return [
            VectorSearchResult(id=vector_id, score=score)
            for vector_id, score in zip(vector_id, result["data"]["Get"]["YourClassName"]["certainty"])
        ]

    def delete_vectors(self, namespace: str):
        if self.client.schema.exists(namespace) is False:
            raise ValueError("Namespace does not exist")
        self.client.schema.delete_class(namespace)
