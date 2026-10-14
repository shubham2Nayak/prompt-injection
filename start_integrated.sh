#!/bin/bash
# PromptTrace Integration Startup Script
# Starts both backend and frontend servers

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}╔════════════════════════════════════════════╗${NC}"
echo -e "${YELLOW}║  PromptTrace - Integrated Startup Script   ║${NC}"
echo -e "${YELLOW}╚════════════════════════════════════════════╝${NC}"
echo ""

# Get backend URL from environment or use default
BACKEND_URL=${BACKEND_URL:-http://localhost:8000}
STREAMLIT_PORT=${STREAMLIT_PORT:-8501}

echo -e "${GREEN}Configuration:${NC}"
echo "  Backend URL: $BACKEND_URL"
echo "  Streamlit Port: $STREAMLIT_PORT"
echo ""

# Check if backend is already running
check_backend() {
    timeout 2 bash -c "curl -s $BACKEND_URL/health > /dev/null 2>&1" && return 0 || return 1
}

# Function to start backend
start_backend() {
    echo -e "${YELLOW}Starting Backend Server...${NC}"
    pushd backend > /dev/null
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || . venv/Scripts/activate
    
    # Install/upgrade dependencies
    pip install -q -r requirements.txt
    
    echo -e "${GREEN}✓ Backend dependencies ready${NC}"
    echo ""
    echo -e "${YELLOW}Backend Server starting on ${BACKEND_URL}${NC}"
    echo "Press Ctrl+C to stop"
    echo ""
    
    python app.py &
    BACKEND_PID=$!
    
    popd > /dev/null
    
    # Wait for backend to start
    echo "Waiting for backend to start..."
    for i in {1..30}; do
        if check_backend; then
            echo -e "${GREEN}✓ Backend is running${NC}"
            return 0
        fi
        sleep 1
    done
    
    echo -e "${RED}✗ Backend failed to start${NC}"
    return 1
}

# Function to start frontend
start_frontend() {
    echo ""
    echo -e "${YELLOW}Starting Frontend Server...${NC}"
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "Creating virtual environment..."
        python -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate 2>/dev/null || . venv/Scripts/activate
    
    # Install/upgrade dependencies
    pip install -q -r requirements_frontend.txt
    
    echo -e "${GREEN}✓ Frontend dependencies ready${NC}"
    echo ""
    echo -e "${YELLOW}Frontend Server starting on http://localhost:${STREAMLIT_PORT}${NC}"
    echo "Press Ctrl+C to stop"
    echo ""
    
    # Set environment variable for backend URL
    export BACKEND_URL=$BACKEND_URL
    
    streamlit run streamlit_app.py --server.port $STREAMLIT_PORT &
    FRONTEND_PID=$!
}

# Main execution
echo -e "${YELLOW}Checking backend status...${NC}"

if check_backend; then
    echo -e "${GREEN}✓ Backend is already running${NC}"
else
    if start_backend; then
        echo -e "${GREEN}✓ Backend started successfully${NC}"
    else
        echo -e "${RED}✗ Failed to start backend${NC}"
        exit 1
    fi
fi

# Start frontend
start_frontend

# Wait for both processes
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  PromptTrace is Running!                   ║${NC}"
echo -e "${GREEN}║  Backend:  $BACKEND_URL${NC}"
echo -e "${GREEN}║  Frontend: http://localhost:${STREAMLIT_PORT}             │"
echo -e "${GREEN}║  Press Ctrl+C to stop                      ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""

# Trap Ctrl+C to kill both processes
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo ''; echo -e '${YELLOW}Shutting down...${NC}'; exit 0" INT TERM

# Wait for both processes
if [ ! -z "$BACKEND_PID" ]; then
    wait $BACKEND_PID
fi

if [ ! -z "$FRONTEND_PID" ]; then
    wait $FRONTEND_PID
fi
