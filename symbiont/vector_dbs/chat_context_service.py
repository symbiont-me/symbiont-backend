from . import reranker
from .vector_store_context import VectorStoreContext
from . import text_splitter

from symbiont.models import CohereTextModels, DocumentPage, Citation
from .models import VectorMetadata, VectorSearchResult
from typing import Union, Tuple, Dict, Optional, List
from .. import logger
from symbiont.mongodb import studies_collection


# DATABASE METHODS


def create_vec_refs_in_db(
    ids: List[str],
    file_identifier: str,
    docs: List[DocumentPage],
    user_id: str,
    study_id: str,
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

    # Create update data for individual fields within vectors.{file_identifier}
    update_data = {f"vectors.{file_identifier}.{id}": data for id, data in vec_data.items()}

    # Use upsert=True to ensure the document is created if it doesn't exist
    studies_collection.update_one(
        {"_id": study_id},
        {"$set": update_data},
        upsert=True,
    )


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

        # TODO move this some place appropriate

    def __truncate_string_by_bytes(self, string: str, num_bytes: int) -> str:
        encoded_string = string.encode("utf-8")
        truncated_string = encoded_string[:num_bytes]
        return truncated_string.decode("utf-8", "ignore")

    def __process_plain_text_and_webpage_resource(self):
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

    # TODO test this
    def add_plaintext_resource(self) -> None:
        self.__process_plain_text_and_webpage_resource()

    def add_web_resource(self) -> None:
        self.__process_plain_text_and_webpage_resource()

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

    # TODO make this private
    resource_adders = {
        "pdf": add_pdf_resource,
        "webpage": add_web_resource,
        "youtube": add_yt_resource,
        "add_plain_text": add_plaintext_resource,
    }

    # TODO this should single Document
    def add_resource(self) -> None:
        if self.resource_type is None:
            raise ValueError("Resource document not provided")
        if self.resource_type not in self.resource_adders:
            raise ValueError("Resource type not supported")
        self.resource_adders[self.resource_type](self)

    # TODO Remove the context from db from here
    # TODO test the removal
    def delete_context(self) -> None:
        self.vector_store_repo.delete_vectors(self.resource_identifier)

    # TODO create Pydantic type for the context
    def rerank_context(self, context: List[Dict[str, str]], query: str) -> Union[Tuple[str, List[Citation]], None]:
        # fixes: cohere.error.CohereAPIError: invalid request: list of documents must not be empty
        if not context:
            return None

        reranked_context = reranker.rerank(
            query=query,
            documents=context,
            top_n=3,
            model=CohereTextModels.COHERE_RERANK_V2,  # TODO the model name should be in a config
        )
        reranked_indices = [r.index for r in reranked_context.results]
        reranked_text = ""
        for text in reranked_context.results:
            reranked_text += text.document.get("text", "")

        # Ensure all required fields are present in the context dictionaries
        citations = [
            Citation(
                text=context[i].get("text", ""),
                source=context[i].get("source", ""),
                page=int(context[i].get("page", 0)),  # if not present, default to 0
            )
            for i in reranked_indices
        ]
        return (reranked_text, citations)

    # TODO document this
    # TODO query does not need to be passed as an arg
    # TODO test and remove query from args
    def get_single_chat_context(self, query: str) -> Optional[Tuple[str, List[Citation]]]:
        try:
            logger.debug(f"Searching vectors for query: {query}")
            results = self.vector_store_repo.search_vectors(namespace=self.resource_identifier, query=query, limit=10)
            ids = [result.id for result in results]
            logger.debug(f"Found {len(ids)} results: {ids}")

            vectors_metadata_from_db = get_vec_refs_from_db(self.study_id, self.resource_identifier, ids)
            logger.debug(f"Found {len(vectors_metadata_from_db)} vectors from db: {vectors_metadata_from_db}")

            logger.debug("Reranking")
            # No need to call dict() on each item
            vectors_metadata_dicts = vectors_metadata_from_db
            # TODO fix this type error if possible or ignore
            # @dev important! this is working as is, so make sure it works if the type error is fixed
            reranked_context = self.rerank_context(vectors_metadata_dicts, query)

            return reranked_context
        except Exception as e:
            logger.error(f"Error in get_single_chat_context: {e}")
            return None

    def get_combined_chat_context(self, query: str) -> Optional[Tuple[str, List[Citation]]]:
        try:
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

                vecs = self.vector_store_repo.search_vectors(namespace=self.resource_identifier, query=query, limit=10)

                combined_vecs.extend(vecs)

            # for each vec get metadata from the db
            ids = [result.id for result in combined_vecs]

            vectors_metadata_from_db: List[VectorMetadata] = []
            for resource in all_resource_identifiers:
                vectors_metadata_from_db.extend(get_vec_refs_from_db(self.study_id, resource, ids))

            # Ensure each dictionary has a "text" key
            # No need to call dict() on each item
            vectors_metadata_dicts = [
                {**vec, "text": vec.get("text", "")} if isinstance(vec, dict) else vec
                for vec in vectors_metadata_from_db
            ]
            # TODO fix this type error if possible or ignore
            # @dev important! this is working as is so make sure it works if the type error is fixed
            reranked_context = self.rerank_context(vectors_metadata_dicts, query)
            return reranked_context
        except Exception as e:
            logger.error(f"Error in get_combined_chat_context: {e}")
            return None
