from typing import Optional
from firebase_admin import firestore
from google.cloud.firestore import DocumentSnapshot
from google.cloud.firestore_v1 import ArrayUnion
from fastapi import HTTPException
from ..models import StudyResource
from .. import logger


class StudyService:
    def __init__(self, user_uid: str, study_id=None):
        self.user_uid = user_uid
        self.study_id = study_id
        self.db = firestore.client()

    def get_document_dict(self) -> Optional[dict]:
        documents_stream = (
            self.db.collection("studies").where("userId", "==", self.user_uid).stream()
        )
        for document in documents_stream:
            if document.id == self.study_id:
                return document.to_dict()
        return None

    def get_document_ref(self):
        documents_stream = (
            self.db.collection("studies").where("userId", "==", self.user_uid).stream()
        )
        for document in documents_stream:
            if document.id == self.study_id:
                return self.db.collection("studies").document(self.study_id)
        return None

    def get_document_snapshot(self) -> Optional[DocumentSnapshot]:
        doc_ref = self.db.collection("studies").document(self.study_id)
        doc_snapshot = doc_ref.get()
        if doc_snapshot.exists:
            return doc_snapshot
        return None

    def add_resource_to_db(self, study_resource: StudyResource):
        study_ref = self.get_document_ref()
        if study_ref is None:
            raise HTTPException(status_code=404, detail="No such document!")
        study_ref.update({"resources": ArrayUnion([study_resource.model_dump()])})

        logger.info(f"Resource added to study {study_resource}")
        return {
            "message": "Resource added successfully",
            "resource": study_resource.model_dump(),
        }

    def set_llm_settings(self, settings):
        doc_ref = self.db.collection("users").document(self.user_uid)
        if not doc_ref:
            return {"error": "User not found", "status_code": 404}
        doc_ref.set(
            {"settings": {"llm_name": settings.llm_name, "api_key": settings.api_key}}
        )
        return {"message": "LLM settings updated"}


class UserService:
    def __init__(self, user_uid: str):
        self.user_uid = user_uid
        self.db = firestore.client()

    def remove_study_from_user(self, study_id):
        user_ref = self.db.collection("users").document(self.user_uid)
        user_ref.update({"studies": firestore.ArrayRemove([study_id])})
        return {"message": "Study removed from user"}

    def create_vec_ref_in_db(self, md5_hash, metadata):
        vec_ref = self.db.collection("users").document(self.user_uid)
        current_data = vec_ref.get().to_dict()
        if "vectors" not in current_data:
            current_data["vectors"] = {}
        current_data["vectors"][md5_hash] = metadata
        vec_ref.set(current_data)
        return vec_ref

    def get_vector_data_from_db(self, md5_hash):
        vec_ref = self.db.collection("users").document(self.user_uid)
        vec_data = vec_ref.get().to_dict()
        return vec_data["vectors"][md5_hash]


# Usage example:
# study_service = StudyService(user_uid="user123", study_id="study456")
# study_dict = study_service.get_document_dict()
