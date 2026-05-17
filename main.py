"""
main.py - FastAPI service for SHL Assessment Advisor.

Endpoints:
  GET  /health  -> {"status": "ok"}
  POST /chat    -> AgentResponse
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from agent import run_agent
from retriever import retriever

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Lifespan: warm up retriever at startup ────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Warming up retriever and embedding model...")
    try:
        retriever.load()
        log.info("Retriever ready.")
    except Exception as e:
        log.error(f"Retriever load failed: {e}")
    yield
    log.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SHL Assessment Advisor",
    description="Conversational agent for SHL Individual Test Solutions",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Pydantic models (schema is non-negotiable per spec) ───────────────────────

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=1)

    @field_validator("messages")
    @classmethod
    def must_have_user_turn(cls, msgs):
        roles = [m.role for m in msgs]
        if "user" not in roles:
            raise ValueError("messages must contain at least one user turn")
        return msgs


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class AgentResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]  # empty list when clarifying
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=AgentResponse)
def chat(request: ChatRequest):
    # Convert to plain dicts for agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Guard: cap at 8 turns total (per spec)
    total_turns = len(messages)
    if total_turns > 8:
        messages = messages[-8:]
        log.warning("Conversation truncated to last 8 turns")

    try:
        result = run_agent(messages)
    except Exception as e:
        log.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    # Validate and build response
    recs = []
    for r in result.get("recommendations", []):
        if isinstance(r, dict) and r.get("name") and r.get("url"):
            recs.append(
                Recommendation(
                    name=r["name"],
                    url=r["url"],
                    test_type=r.get("test_type", ""),
                )
            )

    return AgentResponse(
        reply=result.get("reply", ""),
        recommendations=recs,
        end_of_conversation=bool(result.get("end_of_conversation", False)),
    )
