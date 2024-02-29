import firebase_admin
from dotenv import load_dotenv
import os
from pinecone import Pinecone
import time


load_dotenv()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       FIREBASE INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if not firebase_admin._apps:
    cred = firebase_admin.credentials.Certificate("src/serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {"storageBucket": "symbiont-e7f06.appspot.com"})
    firebase_admin.get_app()
    print("Firebase initialized")


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       PINECONE INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index = os.getenv("PINECONE_INDEX")
pinecone_endpoint = os.getenv("PINECONE_API_ENDPOINT")
pc = Pinecone(api_key=pinecone_api_key, endpoint=pinecone_endpoint)
pc_index = pc.Index("symbiont-me")
# wait for index to be initialized
while not pc.describe_index("symbiont-me").status["ready"]:
    time.sleep(1)

if pc_index is None:
    raise Exception("Pinecone index not found")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       OPENAI INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
openai_api_key = os.getenv("OPENAI_API_KEY")
