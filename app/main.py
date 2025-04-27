# Entry point API server
from fastapi import FastAPI
from app.agent.agent import Agent
from app.schemas.agent_schema import AgentRequest, AgentResponse

app = FastAPI()
agent = Agent()

@app.post("/generate-report", response_model=AgentResponse)
async def generate_report(request: AgentRequest):
    response = agent.handle_request(request)
    return response
