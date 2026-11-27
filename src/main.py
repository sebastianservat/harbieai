from fastapi import FastAPI
from llm import get_llm
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage

SYSTEM_PROMPT = (
    "You are Hub Agent, Harbour Bank's assistant for everything related to leave. "
    "Answer clearly and briefly. If you don't know something, say so."
)

app = FastAPI(title="Harbie API", description="API for Harbie AI application", version="1.0.0")

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat")
def chat(request: ChatRequest) -> ChatResponse:
    reply = get_llm().invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(request.message)])
    return ChatResponse(answer=reply.text)