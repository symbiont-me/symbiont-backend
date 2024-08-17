import firebase_admin
import os
import json
import base64
from .. import logger
import time

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       FIREBASE INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def init_firebase():
    """Initializes the Firebase app using the credentials stored in the FIREBASE_CREDENTIALS environment variable.
    The credentials are base64 encoded and stored in the FIREBASE_CREDENTIALS environment variable.
    The storage bucket is stored in the FIREBASE_STORAGE_BUCKET environment variable.
    """

    if not firebase_admin._apps:
        base_64_cred = os.getenv("FIREBASE_CREDENTIALS")
        storage_bucket = os.getenv("FIREBASE_STORAGE_BUCKET")
        if base_64_cred is None:
            raise KeyError("FIREBASE_CREDENTIALS environment variable not set")
        cred_json = base64.b64decode(base_64_cred).decode("utf-8")
        cred_dict = json.loads(cred_json)
        start_time = time.time()
        cred = firebase_admin.credentials.Certificate(cred_dict)
        # TODO remove storage bucket as not needed after mongo is implemented
        firebase_admin.initialize_app(cred, {"storageBucket": storage_bucket})
        firebase_admin.get_app()
        elapsed_time = time.time() - start_time
        logger.info(f"Firebase initialized in {elapsed_time} seconds")


init_firebase()
