from typing import Optional
from firebase_admin import firestore
from google.cloud.firestore import DocumentSnapshot

"""
These functions are used to ensure that only users with the correct permissions can access the data.
DON'T LIKE THIS
"""


def get_document_dict(
    collection_name: str, field_name: str, field_value: str, document_id: str
) -> Optional[dict]:
    """
    Retrieves a document from Firestore based on the given collection name, field name, field value, and document ID.

    Args:
        collection_name (str): The name of the Firestore collection.
        field_name (str): The name of the field to filter on.
        field_value (str): The value of the field to filter on.
        document_id (str): The ID of the document to retrieve.

    Returns:
        Optional[dict]: The dictionary representation of the retrieved document, or None if the document is not found.
    """
    db = firestore.client()
    documents_stream = (
        db.collection(collection_name).where(field_name, "==", field_value).stream()
    )
    for document in documents_stream:
        if document.id == document_id:
            return document.to_dict()
    return None


def get_document_ref(
    collection_name: str, field_name: str, field_value: str, document_id: str
):
    """
    Retrieves a document reference from Firestore based on the provided collection name, field name, field value, and document ID.

    Args:
        collection_name (str): The name of the collection in Firestore.
        field_name (str): The name of the field to filter the documents.
        field_value (str): The value of the field to filter the documents.
        document_id (str): The ID of the document to retrieve.

    Returns:
        firestore.DocumentReference: The document reference if found, None otherwise.
    """
    db = firestore.client()
    documents_stream = (
        db.collection(collection_name).where(field_name, "==", field_value).stream()
    )
    for document in documents_stream:
        if document.id == document_id:
            return db.collection(collection_name).document(document_id)
    return None


def get_document_snapshot(
    collection: str, document_id: str
) -> Optional[DocumentSnapshot]:
    """
    Fetches a document snapshot from Firestore. If the document exists, returns its data,
    otherwise returns None.

    :param collection: The name of the collection.
    :param document_id: The ID of the document to fetch.
    :return: The document data if exists, otherwise None.
    """
    db = firestore.client()
    doc_ref = db.collection(collection).document(document_id)
    doc_snapshot = doc_ref.get()
    if doc_snapshot.exists:
        return doc_snapshot
    return None
