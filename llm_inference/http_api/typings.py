from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, List, Set

class ChatContext(BaseModel):
    """
    Every message in the history and ChatContext contains these fields
    """
    message: str = Field(..., alias="Message")

    class Config:
        "Auxiliary class for Pydantic parsing of JSONs serialized by .Net apps"
        populate_by_name = True
