# TODO update to get information from .env file
from pinecone import Pinecone
import os
import argparse

from dotenv import load_dotenv


def main():
    load_dotenv()

    if not os.getenv("PINECONE_API_KEY"):
        raise Exception("PINECONE_API_KEY is not set")

    api_key = os.getenv("PINECONE_API_KEY")
    region = "us-west-2"

    parser = argparse.ArgumentParser()
    parser.add_argument("index_name", help="Name of the index to delete")
    args = parser.parse_args()

    pc = Pinecone(api_key=api_key, region=region)
    index = pc.Index(args.index_name)
    namespaces = []
    if index:
        res = index.describe_index_stats()
        namespaces = list(res.get("namespaces", {}).keys())

    for ns in namespaces:
        try:
            if index:
                index.delete(
                    delete_all=True,
                    namespace=ns,
                )
                print(f"Deleted namespace {ns}")
        except Exception as e:
            print(f"Error deleting namespace {ns}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
