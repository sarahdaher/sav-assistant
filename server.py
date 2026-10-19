
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

import agent

app = FastAPI(title="Assistant SAV")

class ChatRequest(BaseModel):
    message: str

def current_user(x_user: str | None) -> str:
    """Authentification simulee : le front envoie le nom dans l'en-tête
    X-User (aka token de session)."""
    if x_user not in agent.USERS:
        raise HTTPException(status_code=401, detail="Utilisateur inconnu")
    return x_user


@app.get("/")
def index():
    return FileResponse("index.html")


@app.get("/api/users")
def users():
    return agent.USERS

@app.post("/api/chat")
def chat(req: ChatRequest, x_user: str = Header()):
    user = current_user(x_user)
    return agent.ask(user, req.message)


@app.post("/api/reset")
def reset(x_user: str | None = Header(default=None)):
    agent.reset(current_user(x_user))
    return {"ok": True}
