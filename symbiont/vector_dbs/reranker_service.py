import cohere
import os
from .. import logger


# @note Currently only supporting Cohere
class RerankerService:
    def __init__(self, reranker_name: str):
        self.api_key = os.getenv("RERANKER_API_KEY")
        if self.api_key is None:
            raise ValueError("Please set the RERANKER_API_KEY environment variable")
        self.reranker_name = reranker_name
        self.reranker = self._init_reranker()

    def _init_reranker(self):
        if self.reranker_name == "cohere":
            logger.info("Reranker: Cohere")
            return cohere.Client(api_key=self.api_key)
        else:
            raise ValueError(f"Reranker {self.reranker_name} not supported")
