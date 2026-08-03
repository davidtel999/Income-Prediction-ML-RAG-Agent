from fastapi import FastAPI
from starlette.responses import RedirectResponse
from pydantic import BaseModel
from src.agent import chat

class ChatRequest(BaseModel):
    message: str  
    session_id: str   

class ChatResponse(BaseModel):
    response: str   

app = FastAPI(title="Income Agent", version="1.0.0")

@app.get("/")
def root():
    return RedirectResponse(url="/docs")

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest) -> ChatResponse:
    response = chat(message=request.message, session_id=request.session_id)
    return ChatResponse(response=response)