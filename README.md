# SHL Assessment Advisor

Conversational agent that recommends SHL Individual Test Solutions through multi-turn dialogue.

## Architecture

```
User → POST /chat → FastAPI → Agent → Groq Llama 3.3
                                 ↓
                           FAISS Retriever (sentence-transformers)
                                 ↓
                           SHL Catalog JSON
```

## Local Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set Groq API key
Get a free key at https://console.groq.com/
```bash
export GROQ_API_KEY=your_key_here
```

### 3. Build catalog
```bash
# Option A: Use seed catalog (recommended, fast)
python scripts/build_seed_catalog.py

# Option B: Scrape live (may be blocked by SHL)
python scripts/scrape_catalog.py
```

### 4. Build FAISS index
```bash
python scripts/build_index.py
```

### 5. Start server
```bash
uvicorn main:app --reload --port 8000
```

### 6. Test
```bash
# Health check
curl http://localhost:8000/health

# Chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "I am hiring a Java developer"}]}'

# Run test suite
python test_agent.py
```

## Deploy to Render (Free)

1. Push to GitHub
2. Create new Render Web Service → connect your repo
3. Set environment variable: `GROQ_API_KEY=your_key`
4. Render uses `render.yaml` automatically — build + start commands are set

> Note: First deploy will be slow (~3 min) as it builds the FAISS index.
> The `/health` endpoint allows up to 2 min for cold start.

## API Reference

### GET /health
```json
{"status": "ok"}
```

### POST /chat
Request:
```json
{
  "messages": [
    {"role": "user", "content": "Hiring a Java developer"},
    {"role": "assistant", "content": "..."},
    {"role": "user", "content": "Mid-level, 4 years experience"}
  ]
}
```

Response:
```json
{
  "reply": "Here are 5 assessments for a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r", "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Agent Behaviour

| Situation | Behaviour |
|-----------|-----------|
| Vague query | Asks clarifying question, empty recommendations |
| Sufficient context | Returns 1–10 ranked recommendations |
| Mid-conversation refinement | Updates shortlist, honours all constraints |
| Comparison question | Answers from catalog data only |
| Off-topic / injection | Refuses with explanation |

## Project Structure

```
shl-agent/
├── main.py                    # FastAPI app + endpoints
├── agent.py                   # Gemini agent + intent logic
├── retriever.py               # FAISS semantic search
├── requirements.txt
├── render.yaml                # Render deployment config
├── start.sh                   # Startup script
├── test_agent.py              # Test suite
├── data/
│   ├── catalog.json           # SHL product catalog
│   ├── index.faiss            # FAISS vector index (built at startup)
│   └── index_meta.json        # Index metadata
└── scripts/
    ├── build_seed_catalog.py  # Generate catalog from known products
    ├── build_index.py         # Build FAISS index from catalog
    └── scrape_catalog.py      # Live scraper (optional)
```
