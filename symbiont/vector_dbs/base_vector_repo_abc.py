from abc import ABC, abstractmethod
from typing import List
from ..models import DocumentPage
from .models import VectorSearchResult


# TODO move to models


class BaseVectorRepository(ABC):
    # @abstractmethod
    # def create_collection(self, collection_name: str, vector_size: int, distance: str):
    #     pass

    # TODO explain what the docs are supposed to be
    @abstractmethod
    def upsert_vectors(self, namespace: str, docs: List[DocumentPage]) -> List:
        pass

    @abstractmethod
    def search_vectors(self, namespace: str, query: str, limit: int) -> List[VectorSearchResult]:
        pass

    # removes the vectors associated with the resource
    @abstractmethod
    def delete_vectors(self, namespace: str):
        pass
