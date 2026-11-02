import os
import time
import asyncio
import traceback
from collections import defaultdict, deque

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tavily import TavilyClient

# ---------- Settings (all come from environment variables) ----------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "").strip()

# Your frontend URL(s), comma separated. Example:
# ALLOWED_ORIGINS=https://book-ai.pages.dev,https://www.yourdomain.com
ALLOWED_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()]

MAX_QUESTION_LENGTH = 300
RATE_LIMIT_REQUESTS = 5      # requests allowed...
RATE_LIMIT_WINDOW = 60       # ...per this many seconds, per visitor

MODELS = [
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
]

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ResearchRequest(BaseModel):
    question: str


# ---------- Simple per-visitor rate limiter ----------
_hits = defaultdict(deque)

def check_rate_limit(request: Request):
    forwarded = request.headers.get("x-forwarded-for", "")
    ip = forwarded.split(",")[0].strip() if forwarded else (request.client.host if request.client else "unknown")
    now = time.time()
    q = _hits[ip]
    while q and now - q[0] > RATE_LIMIT_WINDOW:
        q.popleft()
    if len(q) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(status_code=429, detail="Too many requests. Please wait a minute and try again.")
    q.append(now)


# ---------- Gemini call with retries and model fallback ----------
def call_gemini_rest(prompt_text: str) -> str:
    headers = {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}
    payload = {"contents": [{"parts": [{"text": prompt_text}]}]}
    last_error = None

    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        for attempt in range(3):
            try:
                res = requests.post(url, headers=headers, json=payload, timeout=60)
            except requests.RequestException as e:
                last_error = f"{model}: {e}"
                time.sleep(2 ** attempt)
                continue

            if res.ok:
                parts = res.json()["candidates"][0]["content"]["parts"]
                answer = "".join(p.get("text", "") for p in parts if not p.get("thought", False))
                return answer or "".join(p.get("text", "") for p in parts)

            last_error = f"{model} ({res.status_code})"
            print(f"Gemini warning: {last_error} {res.text[:200]}", flush=True)
            if res.status_code in (429, 500, 503, 504):
                time.sleep(2 ** attempt)
                continue
            break  # 400/403/404: try the next model

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


# ---------- Routes ----------
@app.get("/")
def health():
    return {"status": "ok", "service": "Book AI backend"}


@app.post("/research")
async def conduct_research(body: ResearchRequest, request: Request):
    check_rate_limit(request)

    question = body.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(status_code=400, detail=f"Question too long (max {MAX_QUESTION_LENGTH} characters).")
    if not GEMINI_API_KEY or not TAVILY_API_KEY:
        raise HTTPException(status_code=500, detail="Server is not configured with API keys.")

    try:
        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        search_result = await asyncio.to_thread(
            tavily_client.search, query=question, search_depth="basic", max_results=3
        )

        context = ""
        for r in search_result.get("results", []):
            context += f"\nSource: {r.get('title')}\nContext: {r.get('content')}\n"

        system_prompt = (
            "You are an elite academic textbook assistant. Based on the provided context, "
            "write an extremely descriptive, comprehensive, and well-explained chapter layout. "
            "Do not invent facts. Break it down using these exact sections:\n"
            "1. Executive Summary\n"
            "2. Deep Dive Breakdown\n"
            "3. Key Concepts Explained"
        )
        full_prompt = f"{system_prompt}\n\nStudent Query: {question}\n\nSearch Context Data:\n{context}"

        answer = await asyncio.to_thread(call_gemini_rest, full_prompt)
        return {"answer": answer}
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=503, detail="The AI service is busy right now. Please try again in a moment.")
