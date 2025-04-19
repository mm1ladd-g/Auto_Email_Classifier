# app/schemas.py
from pydantic import BaseModel, Field
from typing import Dict, Literal


class EmailRequest(BaseModel):
    email: str = Field(min_length=10)


class EmailResponse(BaseModel):
    category: Literal["support", "sales", "partnership", "spam"]
    probabilities: Dict[str, float]
