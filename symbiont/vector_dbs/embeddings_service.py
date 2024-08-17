from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter


from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
import os
from langchain_voyageai import VoyageAIEmbeddings
from symbiont.models import EmbeddingModels
from .. import logger


class EmbeddingsService:
    supported_embeddings_models = [
        "text-embedding-3-large",
        "text-embedding-3-small",
        "text-embedding-ada-002",
        "embedding-001",
        "voyage-large-2",
        "voyage-2",
        "voyage-lite-02-instruct",
        "BAAI/bge-base-en",
    ]

    def __init__(self, configs):
        self.configs = configs
        self.embeddings_model, self.text_splitter, self.nltk_text_splitter = self.init_embeddings_settings()

    def init_openai_model(self, model_name: str):
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key is None:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")

        if model_name not in self.supported_embeddings_models:
            raise ValueError(f"Model name {model_name} is not supported")
        return OpenAIEmbeddings(
            model=EmbeddingModels.OPENAI_TEXT_EMBEDDING_3_SMALL,
            dimensions=1536,
            api_key=openai_api_key,
        )

    def init_huggingface_model(self, model_name: str):
        if model_name not in self.supported_embeddings_models:
            raise ValueError(f"Model {model_name} is not supported")
        logger.info("This is a HuggingFace model, it will run locally")
        model_kwargs = {"device": "cpu"}
        encode_kwargs = {"normalize_embeddings": True}
        return HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
        )

    def init_voyage_model(self, model_name: str):
        api_key = os.getenv("EMBEDDINGS_MODEL_API_KEY")
        if api_key is None:
            raise ValueError("Please set the VOYAGE_API_KEY environment variable")

        if model_name not in self.supported_embeddings_models:
            raise ValueError(f"Model name {model_name} is not supported")

        return VoyageAIEmbeddings(model=model_name, voyage_api_key=api_key, batch_size=1)  # type: ignore

    def init_embeddings_model(self):
        if "bge-base-en" in self.configs.embeddings_model:
            return self.init_huggingface_model(self.configs.embeddings_model)
        if "voyage" in self.configs.embeddings_model:
            return self.init_voyage_model(self.configs.embeddings_model)
        if "text-embedding" in self.configs.embeddings_model:
            return self.init_openai_model(self.configs.embeddings_model)

    def init_embeddings_settings(self):
        nltk_text_splitter = NLTKTextSplitter()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=0)
        embeddings_model = self.init_embeddings_model()
        logger.info(
            f"Embeddings model: {embeddings_model.model}"
        )  # TODO .model will probably give errors for huggingface
        return embeddings_model, text_splitter, nltk_text_splitter
