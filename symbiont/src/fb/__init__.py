import firebase_admin
import os
import json
import base64

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       FIREBASE INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if not firebase_admin._apps:
    base_64_cred = os.getenv("FIREBASE_CREDENTIALS")
    if not base_64_cred:
        raise ValueError("FIREBASE_CREDENTIALS environment variable not set")
    cred_json = base64.b64decode(base_64_cred).decode("utf-8")
    cred_dict = json.loads(cred_json)
    cred = firebase_admin.credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred, {"storageBucket": "symbiont-e7f06.appspot.com"})
    firebase_admin.get_app()
    print("Firebase initialized")
