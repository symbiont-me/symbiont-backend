import firebase_admin

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#       FIREBASE INIT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if not firebase_admin._apps:
    cred = firebase_admin.credentials.Certificate("src/serviceAccountKey.json")
    firebase_admin.initialize_app(cred, {"storageBucket": "symbiont-e7f06.appspot.com"})
    firebase_admin.get_app()
    print("Firebase initialized")
