# from fastapi import FastAPI, File, UploadFile
# from firebase_admin import initialize_app, storage, credentials

# cred = credentials.Certificate("src/serviceAccountKey.json")
# initialize_app(cred, {"storageBucket": "your_bucket_link_without_gs://"})


# def save_file(file: UploadFile):
#     bucket = storage.bucket()
#     blob = bucket.blob(file.filename)
#     file_content = file.file.read()
#     blob.upload_from_string(file_content)

#     public_url = f"https://storage.googleapis.com/{bucket.name}/{blob.name}"
#     return public_url
