# Input/Output Schema
from pydantic import BaseModel
from typing import Optional

class AgentRequest(BaseModel):
    prompt: str

class AgentResponse(BaseModel):
    report: str
    plan: Optional[str]
    refined_plan: Optional[str]
    file_context: Optional[str]
    web_context: Optional[str]
