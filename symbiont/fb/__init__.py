import firebase_admin
import os
import json
import base64
from .. import logger
import time
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       FIREBASE INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


if not firebase_admin._apps:
    base_64_cred = os.getenv("FIREBASE_CREDENTIALS")
    storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
    if not base_64_cred:
        raise ValueError("FIREBASE_CREDENTIALS environment variable not set")
    cred_json = base64.b64decode(base_64_cred).decode("utf-8")
    cred_dict = json.loads(cred_json)
    start_time = time.time()
    cred = firebase_admin.credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred, {"storageBucket": storage_bucket})
    firebase_admin.get_app()
    elapsed_time = time.time() - start_time
    logger.info(f"Firebase initialized in {elapsed_time} seconds")
