"""
TODO: Needs to be updated to make methods async where appropriate
TODO: The whole file needs to be refactored.
TODO: Param and Return types need to be specified
TODO: Clear documentation needs to be added, with regards to purpose of the classes and the file
NOTE: The main interfaces should be maintained, because it allows for easy addition of new vector databases
"""

from qdrant_client import QdrantClient

from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_community.document_loaders import TextLoader
from typing import List
from qdrant_client.http.models import Batch, PointStruct
import uuid
from langchain_openai import OpenAIEmbeddings
import numpy as np
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional
from pydantic import BaseModel
from langchain.text_splitter import NLTKTextSplitter, RecursiveCharacterTextSplitter

from langchain_voyageai import VoyageAIEmbeddings

from ..models import EmbeddingModels, CohereTextModels, DocumentPage, Citation
from .. import logger
from ..mongodb import studies_collection
import cohere
from typing import Union, Tuple

# 1. Set up a Milvus client
# from env read name of the vector store
# from env read name of the embeddings model


cohere_api_key = os.getenv("CO_API_KEY")

co = cohere.Client(api_key=cohere_api_key or "")


class Metadata(BaseModel):
    source: str
    page: Optional[str]


class Document(BaseModel):
    page_content: str
    metadata: Metadata


class PineconeRecord(BaseModel):
    id: str
    values: List[float]
    metadata: Metadata


class PineconeResult(BaseModel):
    id: str
    score: float
    values: List


class PineconeResults(BaseModel):
    matches: List[PineconeResult]


class VectorSearchResult(BaseModel):
    id: str
    score: float


class VectorRef(BaseModel):
    source: str
    page: str
    text: str


def create_ids(n):
    return [str(uuid.uuid4()) for _ in range(n)]


class BaseVectorRepository(ABC):
    # @abstractmethod
    # def create_collection(self, collection_name: str, vector_size: int, distance: str):
    #     pass

    # TODO explain what the docs are supposed to be
    @abstractmethod
    def upsert_vectors(self, namespace: str, docs) -> List:  # docs can be either a list or None
        pass

    @abstractmethod
    # TODO return type should be List[VectorSearchResult]
    # TODO modify the method in other classes

    def search_vectors(self, namespace: str, query: str, limit: int) -> List[VectorSearchResult]:
        pass

    # removes the vectors associated with the resource
    @abstractmethod
    def delete_vectors(self, namespace: str):
        pass

    # TODO remove this and the one below
    def create_embeddings(self):
        pass

    # TODO add openai and other models
    def init_embeddings_model(self):
        model_name = "BAAI/bge-base-en"
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )


# TODO try except for the env variables and remove the default values
class VectorStoreSettings:
    def __init__(self):
        self.vector_store = os.getenv("VECTOR_STORE")
        self.vector_store_url = os.getenv("VECTOR_STORE_URL")
        self.vector_store_port = os.getenv("VECTOR_STORE_PORT")
        self.vector_store_dimension = os.getenv("VECTOR_STORE_DIMENSION")
        self.vector_store_distance = os.getenv("VECTOR_STORE_DISTANCE")
        self.vector_store_token = os.getenv("VECTOR_STORE_TOKEN")  # should be optional


nltk_text_splitter = NLTKTextSplitter()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)

voyage_api_key = os.getenv("VOYAGE_API_KEY")

# TODO fix type issue
embeddings_model = VoyageAIEmbeddings(voyage_api_key=voyage_api_key, model=EmbeddingModels.VOYAGEAI_2_LARGE)


# class EmbeddingsService:
#     def init_embeddings_model(self, model_name: str):
#         if model_name == "bge-base-en":
#             model_name = "BAAI/bge-base-en"
#             model_kwargs = {"device": "cpu"}
#             encode_kwargs = {"normalize_embeddings": True}
#             return HuggingFaceBgeEmbeddings(
#                 model_name=model_name,
#                 model_kwargs=model_kwargs,
#                 encode_kwargs=encode_kwargs,
#             )
#
#     def __init__(self, model_name: str):
#         self.model = self.init_embeddings_model(model_name)
#
#     def create_docs_embeddings(self, docs):
#         docs_for_embedding = [doc.page_content for doc in docs]
#         if self.model is not None:
#             embeddings = [self.model.embed_query(text) for text in docs_for_embedding]
#             return embeddings
#         else:
#             raise ValueError("Embeddings model not initialized")
#
#
# embeddings_service = EmbeddingsService("bge-base-en")
vector_store_settings = VectorStoreSettings()


# TODO
# studies_repo = ...


class QdrantRepository(BaseVectorRepository):
    def __init__(self):
        self.dimension = vector_store_settings.vector_store_dimension
        self.distance = vector_store_settings.vector_store_distance
        self.client = QdrantClient(
            url=f"{vector_store_settings.vector_store_url}:{vector_store_settings.vector_store_port}",
            port=6333,
            api_key=vector_store_settings.vector_store_token,
        )

        print("Connected to Qdrant")

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
        if embeddings_model.model is None:
            raise ValueError("Embeddings model not initialized")
        vectorised_query = embeddings_model.embed_query(query)
        results = self.client.search(collection_name=namespace, query_vector=vectorised_query, limit=limit)
        transformed_results = self.__transform_search_results(results)
        logger.info(f"Found {len(transformed_results)} results")
        return transformed_results

    def delete_vectors(self, namespace: str) -> None:
        self.client.delete_collection(collection_name=namespace)
        logger.info(f"Deleted: Vectors for {namespace}")


vector_store_repos = {
    "qdrant": QdrantRepository,
}


class VectorStoreContext:
    def __init__(self):
        self.vector_store = vector_store_settings.vector_store
        if self.vector_store not in vector_store_repos:
            raise ValueError("Vector store not supported")
        if vector_store_settings.vector_store is None:
            raise ValueError("Set the Vector Store name")
        # NOTE this instiates the vector store repo using the object
        self.vector_store_repo = vector_store_repos[vector_store_settings.vector_store]()


def create_vec_refs_in_db(
    ids: List[str], file_identifier: str, docs: List[DocumentPage], user_id: str, study_id: str
) -> None:
    # TODO check user access
    if len(ids) != len(docs):
        raise ValueError("The lengths of 'ids' and 'docs' must be the same")

    vec_data = {}
    for id, doc in zip(ids, docs):
        vec_data[id] = {
            "source": file_identifier,
            "page": doc.metadata.get("page"),
            "text": doc.page_content,
        }

    logger.debug(f"Creating Vectors in DB: {vec_data}")
    studies_collection.update_one(
        {"_id": study_id},
        {"$set": {f"vectors.{file_identifier}": vec_data}},
    )


class VectorMetadata(BaseModel):
    source: str
    page: str
    text: str


# TODO this should be part of a db repo or service
def get_vec_refs_from_db(study_id: str, file_identifier: str, ids: List[str]) -> List[VectorMetadata]:
    logger.info(f"Fetching Vectors from {study_id}")

    logger.info("Fetching Vectors from DB")
    results = []
    study = studies_collection.find_one({"_id": study_id})
    if study is None:
        raise ValueError("Study not found")
    study_vectors = study.get("vectors", {})
    file_vectors = study_vectors.get(file_identifier, {})

    for id in ids:
        vec = file_vectors.get(id, {})
        results.append(vec)
    logger.debug(f"Found {len(results)} vectors")
    # TODO rename: this is the vec data from the db
    return results


# 1. langchain parses a resource and creates a document with a sinlge element called Document
# 2. we have to split the document into smaller chunks
# 3. we have to create embeddings for each chunk
# 4. we have to use the standard return object
class ChatContextService(VectorStoreContext):
    def __init__(
        self,
        resource_doc=None,
        resource_identifier="",  # TODO make this a required param for initialisation
        resource_type=None,
        user_id: str = "",
        user_query: str = "",
        study_id: str = "",  # TODO this should be required
    ):
        super().__init__()
        self.user_id = user_id
        self.resource_identifier = resource_identifier
        self.user_query = user_query
        self.resource_doc = resource_doc
        self.resource_type = resource_type
        self.study_id = study_id

    def __truncate_string_by_bytes(self, string: str, num_bytes: int) -> str:
        encoded_string = string.encode("utf-8")
        truncated_string = encoded_string[:num_bytes]
        return truncated_string.decode("utf-8", "ignore")

    # @dev this performs operations on a single pdf document, splite the content and make it into Document Page
    # that can be used by the vector store
    def __parse_pdf_doc(self, pdf_page: DocumentPage) -> List[DocumentPage]:
        page_content = pdf_page.page_content.replace("\n", "")
        page_content = self.__truncate_string_by_bytes(page_content, 10000)
        split_texts = text_splitter.create_documents([page_content])
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": self.resource_identifier,
                    "page": pdf_page.metadata["page"],
                },
                type=pdf_page.type,
            )
            for split_text in split_texts
        ]

        return docs

    def add_pdf_resource(self) -> None:
        if self.resource_doc is None:
            raise ValueError("Resource document not provided")
        docs = []
        for pdf_page in self.resource_doc:
            docs.extend(self.__parse_pdf_doc(pdf_page))

        ids = self.vector_store_repo.upsert_vectors(self.resource_identifier, docs)
        create_vec_refs_in_db(ids, self.resource_identifier, docs, self.user_id, self.study_id)

    def add_web_resource(self) -> None:
        content = getattr(self.resource_doc, "page_content", None)
        if content is None:
            raise ValueError("There is no resource content to be added")
        split_texts = text_splitter.create_documents([content])
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": self.resource_identifier,
                    "page": 0,
                },
            )
            for split_text in split_texts
        ]
        ids = self.vector_store_repo.upsert_vectors(self.resource_identifier, docs)
        create_vec_refs_in_db(ids, self.resource_identifier, docs, self.user_id, self.study_id)

    def add_yt_resource(self) -> None:
        content = getattr(self.resource_doc, "page_content", None)
        if content is None:
            raise ValueError("There is no resource content to be added")
        split_texts = text_splitter.create_documents([content])
        docs = [
            DocumentPage(
                page_content=split_text.page_content,
                metadata={
                    "text": split_text.page_content,
                    "source": self.resource_identifier,
                    "page": 0,
                },
            )
            for split_text in split_texts
        ]
        ids = self.vector_store_repo.upsert_vectors(self.resource_identifier, docs)
        logger.debug(ids)
        create_vec_refs_in_db(ids, self.resource_identifier, docs, self.user_id, self.study_id)

    # TODO move this some place appropriate
    resource_adders = {
        "pdf": add_pdf_resource,
        "webpage": add_web_resource,
        "youtube": add_yt_resource,
    }

    # TODO this should single Document
    def add_resource(self) -> None:
        if self.resource_type is None:
            raise ValueError("Resource document not provided")
        if self.resource_type not in self.resource_adders:
            raise ValueError("Resource type not supported")
        self.resource_adders[self.resource_type](self)

    # TODO Remove the context from db from here
    def delete_context(self) -> None:
        self.vector_store_repo.delete_vectors(self.resource_identifier)

    # TODO add the type for context
    def rerank_context(self, context: List[Dict[str, str]], query: str) -> Union[Tuple[str, List[Citation]], None]:
        # fixes: cohere.error.CohereAPIError: invalid request: list of documents must not be empty
        if not context:
            return None

        reranked_context = co.rerank(
            query=query,
            documents=context,
            top_n=3,
            model=CohereTextModels.COHERE_RERANK_V2,
        )
        reranked_indices = [r.index for r in reranked_context.results]
        citations = [context[i] for i in reranked_indices]
        reranked_text = ""
        for text in reranked_context.results:
            reranked_text += text.document.get("text", "")
        # Convert dict to Citation with page as int
        # @note we just need to make sure the return data type complies
        citations = [Citation(**{**context[i], "page": int(context[i]["page"])}) for i in reranked_indices]
        return (reranked_text, citations)

    # TODO document this
    # TODO query does not need to be passed as an arg
    def get_single_chat_context(self, query: str) -> Optional[Tuple[str, List[Citation]]]:
        results = self.vector_store_repo.search_vectors(namespace=self.resource_identifier, query=query, limit=10)
        ids = [result.id for result in results]
        logger.debug(f"Found {len(ids)} results")
        vectors_metadata_from_db = get_vec_refs_from_db(self.study_id, self.resource_identifier, ids)
        logger.debug(f"Found {len(vectors_metadata_from_db)} vectors from db")
        logger.debug("Reranking")
        vectors_metadata_dicts = [vec.dict() for vec in vectors_metadata_from_db]
        reranked_context = self.rerank_context(vectors_metadata_dicts, query)
        return reranked_context

    def get_combined_chat_context(self, query: str) -> Optional[Tuple[str, List[Citation]]]:
        logger.debug("==========GETTING COMBINED CONTEXT==========")
        # get the identifier for each resource
        study = studies_collection.find_one({"_id": self.study_id})
        logger.debug(f"Fetching Study: {study}")
        if study is None:
            raise ValueError("Study not found")
        resources = study.get("resources", [])

        all_resource_identifiers = [resource.get("identifier") for resource in resources]

        # array of vec ids and scores
        combined_vecs: List[VectorSearchResult] = []
        # combine the context
        for identifier in all_resource_identifiers:
            self.resource_identifier = identifier

            vecs = self.vector_store_repo.search_vectors(
                namespace=self.resource_identifier, query=self.user_query, limit=10
            )

            combined_vecs.extend(vecs)

        # for each vec get metadata from the db
        ids = [result.id for result in combined_vecs]

        vectors_metadata_from_db: List[VectorMetadata] = []
        for resource in all_resource_identifiers:
            vectors_metadata_from_db.extend(get_vec_refs_from_db(self.study_id, resource, ids))
        # rerank the context
        vectors_metadata_dicts = [vec.model_dump() for vec in vectors_metadata_from_db]
        reranked_context = self.rerank_context(vectors_metadata_dicts, query)
        return reranked_context


# TODO remove as not being used
def create_docs(path: str):
    loader = TextLoader(path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs
