from typing import Any
from pydantic import BaseModel

class EvalResult(BaseModel):
    response: str # respone
