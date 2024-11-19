# app/models.py
from pydantic import BaseModel
from typing import Optional

class UserInput(BaseModel):
    user_name: str
    field: str
    experience: str
    years_of_experience: Optional[str] = None

class UserResponse(BaseModel):
    user_response: str

