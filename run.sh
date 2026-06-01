#!/bin/bash
# Runic AI - Remote Training Script
# Usage: ./run.sh

set -e  # Exit on error

echo "=============================================="
echo "  Runic AI - Remote Training Script"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PYTHON_VERSION="3.10"
VENV_NAME="runic_env"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo -e "${YELLOW}📁 Working directory: ${SCRIPT_DIR}${NC}"
echo ""

# Step 1: Check Git
echo -e "${YELLOW}Step 1: Checking Git installation...${NC}"
if ! command -v git &> /dev/null; then
    echo -e "${YELLOW}⚠️  Git not found. Installing...${NC}"
    apt-get update && apt-get install -y git
    echo -e "${GREEN}✓ Git installed${NC}"
else
    echo -e "${GREEN}✓ Git found${NC}"
    git --version
fi
echo ""

# Step 2: Check Python
echo -e "${YELLOW}Step 2: Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed. Please install Python ${PYTHON_VERSION}+${NC}"
    exit 1
fi

PYTHON_PATH=$(which python3)
echo -e "${GREEN}✓ Python found: ${PYTHON_PATH}${NC}"
python3 --version
echo ""

# Step 3: Create virtual environment (if not exists)
echo -e "${YELLOW}Step 2: Setting up virtual environment...${NC}"
if [ ! -d "${VENV_NAME}" ]; then
    echo "Creating virtual environment '${VENV_NAME}'..."
    python3 -m venv ${VENV_NAME}
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Step 4: Activate virtual environment
echo -e "${YELLOW}Step 3: Activating virtual environment...${NC}"
source ${VENV_NAME}/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Step 5: Upgrade pip
echo -e "${YELLOW}Step 4: Upgrading pip...${NC}"
pip install --upgrade pip
echo ""

# Step 6: Install PyTorch (CUDA version for NVIDIA GPUs)
echo -e "${YELLOW}Step 5: Installing PyTorch with CUDA support...${NC}"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Step 7: Install dependencies
echo -e "${YELLOW}Step 6: Installing project dependencies...${NC}"
if [ -f "./train_model/requirements.txt" ]; then
    pip install -r ./train_model/requirements.txt
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}❌ ./train_model/requirements.txt not found!${NC}"
    exit 1
fi
echo ""

# Step 8: Verify installation
echo -e "${YELLOW}Step 7: Verifying installation...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import pandas; print(f'Pandas: {pandas.__version__}')"
python -c "import clearml; print(f'ClearML: {clearml.__version__}')"
echo -e "${GREEN}✓ All imports successful${NC}"
echo ""

# Step 9: Check GPU availability
echo -e "${YELLOW}Step 8: Checking GPU availability...${NC}"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if torch.cuda.is_available(); then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    echo -e "${GREEN}✓ GPU detected${NC}"
else
    echo -e "${YELLOW}⚠️  No GPU detected, training will use CPU (slower)${NC}"
fi
echo ""

# Step 10: Check ClearML configuration
echo -e "${YELLOW}Step 9: Checking ClearML configuration...${NC}"
if [ -z "$CLEARML_API_ACCESS_KEY" ] && [ -z "$CLEARML_API_SECRET_KEY" ]; then
    if [ -f ~/.clearml.conf ]; then
        echo -e "${GREEN}✓ ClearML config found~/.clearml.conf${NC}"
    else
        echo -e "${YELLOW}⚠️  ClearML API keys not set. Training will run without logging.${NC}"
        echo "   Set environment variables or run 'clearml-init'"
    fi
else
    echo -e "${GREEN}✓ ClearML API keys found in environment${NC}"
fi
echo ""

# Step 11: Run training
echo -e "${YELLOW}Step 10: Starting training...${NC}"
echo "=============================================="
echo ""

python train_model.py

# Capture exit code
EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully!${NC}"
else
    echo -e "${RED}❌ Training failed with exit code ${EXIT_CODE}${NC}"
fi
echo "=============================================="

# Deactivate virtual environment
deactivate

exit $EXIT_CODE
