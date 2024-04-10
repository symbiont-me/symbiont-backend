# NOTE: These routes are for testing purposes only

from firebase_admin import firestore
from fastapi import APIRouter, HTTPException
from .. import logger

router = APIRouter()


@router.post("/doc-size-exceeded")
async def simulate_error():
    try:
        db = firestore.client()
        doc_ref = db.collection("studies").document("large_document")
        large_dict = {"key": "a" * 1048577}
        doc_ref.set(large_dict)
        logger.info("Document set")
        return {"status": "done"}
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail="Document size exceeded")
