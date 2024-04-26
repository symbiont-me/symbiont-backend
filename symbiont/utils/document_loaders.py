import fitz  # PyMuPDF
from typing import List
from ..models import DocumentPage


def load_pdf(file, file_identifier) -> List[DocumentPage]:
    docs = []

    doc = fitz.open(file)

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text()
        docs.append(
            DocumentPage(
                page_content=text, metadata={"source": file_identifier, "page": str(page_number), "text": text}
            )
        )

    return docs
