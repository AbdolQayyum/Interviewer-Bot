from pydantic import BaseModel
from typing import Optional


class UserInput(BaseModel):
    candidate_id: str  # Added candidate_id to uniquely identify the candidate
    candidate_name: str
    email: str
    job_title: str
    experience: str
    years_of_experience: Optional[str] = None  # Optional field for more granular experience tracking


class UserResponse(BaseModel):
    user_response: str
