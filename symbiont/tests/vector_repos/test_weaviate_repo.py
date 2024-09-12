import pytest
from unittest.mock import patch
from symbiont.vector_dbs.vector_store_repos.weaviate_repo import (
    upsert_vectors,
    init_weaviate,
)
from symbiont.vector_dbs.vector_store_repos.weaviate_repo import search_vectors
import torch


# Mock embedding model
class MockEmbedding:
    def embed_query(self, text):
        return torch.randn(1536).tolist()


@pytest.fixture
def weaviate_client():
    return init_weaviate()


@pytest.fixture
def embedding():
    return MockEmbedding()


content = """April is the cruellest month, breeding
Lilacs out of the dead land, mixing
Memory and desire, stirring
Dull roots with spring rain.
Winter kept us warm, covering
Earth in forgetful snow, feeding
A little life with dried tubers.
Summer surprised us, coming over the Starnbergersee
With a shower of rain; we stopped in the colonnade,
And went on in sunlight, into the Hofgarten,
And drank coffee, and talked for an hour.
Bin gar keine Russin, stamm’ aus Litauen, echt deutsch.
And when we were children, staying at the archduke’s,
My cousin’s, he took me out on a sled,
And I was frightened. He said, Marie,
Marie, hold on tight. And down we went.
In the mountains, there you feel free.
I read, much of the night, and go south in the winter."""
documents = [
    "A group of vibrant parrots chatter loudly, sharing stories of their tropical adventures.",
    "The mathematician found solace in numbers, deciphering the hidden patterns of the universe.",
    "The robot, with its intricate circuitry and precise movements, assembles the devices swiftly.",
    "The chef, with a sprinkle of spices and a dash of love, creates culinary masterpieces.",
    "The ancient tree, with its gnarled branches and deep roots, whispers secrets of the past.",
    "The detective, with keen observation and logical reasoning, unravels the intricate web of clues.",
    "The sunset paints the sky with shades of orange, pink, and purple, reflecting on the calm sea.",
    "In the dense forest, the howl of a lone wolf echoes, blending with the symphony of the night.",
    "The dancer, with graceful moves and expressive gestures, tells a story without uttering a word.",
    "In the quantum realm, particles flicker in and out of existence, dancing to the tunes of probability.",
]


@pytest.fixture
def docs():
    return documents


query = "Give me some content about the ocean"


@patch(
    "symbiont.vector_dbs.vector_store_repos.weaviate_repo.embedding",
    new_callable=MockEmbedding,
)
def test_upsert_vectors(mock_embedding, weaviate_client, docs):
    namespace = "DocumentSearch"

    # Mock the client methods
    with patch.object(weaviate_client.schema, "create_class") as mock_create_class, patch.object(
        weaviate_client.batch, "configure"
    ) as mock_configure, patch.object(weaviate_client.batch, "add_data_object") as mock_add_data_object:
        result_ids = upsert_vectors(namespace, docs)

        # Assertions
        assert len(result_ids) == len(docs)


def test_search_vectors(capsys):
    # Call the function with actual parameters
    search_vectors("DocumentSearch", query, 25)
