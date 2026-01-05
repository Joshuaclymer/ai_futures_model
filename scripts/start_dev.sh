#!/bin/bash

# Start the frontend and backend development servers
# This script kills any existing sessions before starting new ones

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/app_frontend"
BACKEND_DIR="$PROJECT_ROOT/app_backend"

FRONTEND_PORT=3000
BACKEND_PORT=5329

# Kill process on a given port
kill_port() {
    local port=$1
    local pids=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pids" ]; then
        echo "Killing existing process(es) on port $port..."
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 1
    fi
}

# Kill existing sessions
echo "Checking for existing sessions..."
kill_port $FRONTEND_PORT
kill_port $BACKEND_PORT

# Start backend
echo "Starting backend on port $BACKEND_PORT..."
cd "$BACKEND_DIR"
python3 api.py &
BACKEND_PID=$!

# Start frontend
echo "Starting frontend on port $FRONTEND_PORT..."
cd "$FRONTEND_DIR"
npm run next-dev &
FRONTEND_PID=$!

echo ""
echo "Development servers started:"
echo "  Frontend: http://localhost:$FRONTEND_PORT (PID: $FRONTEND_PID)"
echo "  Backend:  http://localhost:$BACKEND_PORT (PID: $BACKEND_PID)"
echo ""
echo "Press Ctrl+C to stop both servers"

# Handle Ctrl+C to kill both processes
trap "echo ''; echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit 0" SIGINT SIGTERM

# Wait for both processes
wait
