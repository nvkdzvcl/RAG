"""Common schema objects."""

from enum import Enum

from pydantic import BaseModel


class Mode(str, Enum):
    """Supported query modes."""

    STANDARD = "standard"
    ADVANCED = "advanced"
    COMPARE = "compare"


class Citation(BaseModel):
    """Citation object returned with grounded answers."""

    chunk_id: str
    doc_id: str
    source: str
    title: str | None = None
    section: str | None = None
    page: int | None = None
    file_name: str | None = None
    file_type: str | None = None
    uploaded_at: str | None = None
    created_at: str | None = None
    block_type: str | None = None
    ocr: bool | None = None
