#!/bin/bash

# PromptTrace - Startup Script
# Launches both backend and frontend

echo "🚀 PromptTrace Startup Script"
echo "=============================="
echo ""

# Check Python availability
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.8+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"
echo "✅ Environment: $(which python3)"
echo ""

# This platform provides a single managed conda environment, so a local venv
# cannot be created here. The script runs against the active interpreter.
echo "📦 Using the active environment (venv creation is not supported here)"
echo ""

BACKEND_PORT="${API_PORT:-8000}"
BACKEND_PID=""

# A previous run of this script backgrounds the backend with `&`. If its
# terminal is closed the backend survives and keeps every model it loaded
# resident, which silently consumes most of the GPU. Reclaim it before starting.
release_stale_backend() {
    local pids
    pids=$(ss -lptnH "sport = :$BACKEND_PORT" 2>/dev/null |
           grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
    if [ -n "$pids" ]; then
        echo "⚠️  Releasing stale backend on port $BACKEND_PORT (pid: $(echo $pids | tr '\n' ' '))"
        echo "$pids" | xargs -r kill 2>/dev/null
        sleep 3
        echo "$pids" | xargs -r kill -9 2>/dev/null
    fi
}

cleanup() {
    if [ -n "$BACKEND_PID" ] && kill -0 "$BACKEND_PID" 2>/dev/null; then
        kill "$BACKEND_PID" 2>/dev/null
        wait "$BACKEND_PID" 2>/dev/null
    fi
}

# Run the backend and wait until it answers /health (or give up loudly).
start_backend() {
    ( cd backend && exec python -u app.py ) &
    BACKEND_PID=$!

    for _ in $(seq 1 120); do
        if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
            echo "❌ Backend exited during startup. Last log lines:"
            return 1
        fi
        if curl -sf "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then
            echo "✅ Backend ready on http://127.0.0.1:$BACKEND_PORT"
            return 0
        fi
        sleep 2
    done

    echo "❌ Backend did not become healthy within 240s. Last log lines:"
    return 1
}

# Offer options
echo "What would you like to do?"
echo ""
echo "1) Start Backend only"
echo "2) Start Frontend only"
echo "3) Start Both (in separate terminals)"
echo "4) Exit"
echo ""

trap cleanup EXIT INT TERM

read -p "Enter choice (1-4): " choice

case $choice in
    1)
        release_stale_backend
        echo "🚀 Starting Backend Server..."
        ( cd backend && exec python -u app.py )
        ;;
    2)
        echo "🚀 Starting Frontend Dashboard..."
        streamlit run streamlit_app.py
        ;;
    3)
        release_stale_backend
        echo "🚀 Starting Both Services..."
        echo ""
        echo "Starting Backend in background..."
        start_backend || exit 1

        echo ""
        echo "Starting Frontend..."
        streamlit run streamlit_app.py
        ;;
    4)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice"
        exit 1
        ;;
esac
