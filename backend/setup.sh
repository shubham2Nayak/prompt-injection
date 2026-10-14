#!/bin/bash
# PromptTrace Backend - Setup and Startup Script

set -e

echo "🚀 PromptTrace Backend Setup"
echo "=============================="

# Check Python version
echo "📦 Checking Python version..."
python --version || python3 --version
if ! python -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' 2>/dev/null; then
    echo "❌ Python 3.11+ required"
    exit 1
fi
echo "✅ Python version OK"

# Check CUDA (if available)
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "🖥️  GPU Information:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo "✅ CUDA available"
else
    echo "⚠️  CUDA not found - will use CPU (slower)"
fi

# Virtual environment
echo ""
echo "🔧 Setting up virtual environment..."
if [ ! -d "venv" ]; then
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
fi

# Install dependencies
echo ""
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
echo "✅ Dependencies installed"

# Environment configuration
echo ""
echo "⚙️  Checking configuration..."
if [ ! -f ".env" ]; then
    echo "📋 Creating .env from template..."
    cp .env.example .env
    echo "✅ .env created - please edit with your settings"
else
    echo "✅ .env already exists"
fi

# Create data directories
echo ""
echo "📁 Creating data directories..."
mkdir -p data/uploads data/indices data/logs data/cache
echo "✅ Data directories ready"

# Verify imports
echo ""
echo "✔️  Verifying imports..."
python -c "
import torch
import fastapi
import langgraph
import transformers
import sentence_transformers
import faiss
print('✅ All imports successful')
print(f'   PyTorch version: {torch.__version__}')
print(f'   CUDA available: {torch.cuda.is_available()}')
" || exit 1

echo ""
echo "=============================="
echo "✅ Setup Complete!"
echo ""
echo "📖 Next steps:"
echo "   1. Edit .env with your configuration"
echo "   2. Start server: python app.py"
echo "   3. Visit: http://localhost:8000"
echo ""
echo "📚 Documentation: See README.md"
echo ""
