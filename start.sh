#!/usr/bin/env bash
# start.sh - Run on Render or any host
# 1. Build catalog if not present
# 2. Build FAISS index if not present
# 3. Start FastAPI server

set -e

echo "=== SHL Assessment Advisor startup ==="

# Build catalog from seed data if needed
if [ ! -f "data/catalog.json" ]; then
    echo "Building seed catalog..."
    python scripts/build_seed_catalog.py
fi

# Build FAISS index if needed
if [ ! -f "data/index.faiss" ]; then
    echo "Building FAISS index (first run, ~2 min)..."
    python scripts/build_index.py
fi

echo "Starting FastAPI server..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
