from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Literal

AllowedCategory = Literal["support", "sales", "partnership", "spam", "unknown"]


class EmailRequest(BaseModel):
    email: str = Field(min_length=10)

    model_config = ConfigDict(extra="forbid")  # reject unexpected keys


class EmailResponse(BaseModel):
    category: AllowedCategory
    probabilities: Dict[str, float]

    model_config = ConfigDict(extra="forbid")
