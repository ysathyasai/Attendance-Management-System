#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Attendance Management System Setup${NC}"
echo -e "${GREEN}================================${NC}\n"

# Check if Python is installed
echo -e "${YELLOW}[1/5] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8+${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Python found: $(python3 --version)${NC}\n"

# Check if MySQL is installed
echo -e "${YELLOW}[2/5] Checking MySQL installation...${NC}"
if ! command -v mysql &> /dev/null; then
    echo -e "${YELLOW}⚠ MySQL is not installed. You'll need to set it up manually.${NC}"
    echo -e "${YELLOW}   Visit: https://dev.mysql.com/downloads/mysql/${NC}\n"
else
    echo -e "${GREEN}✓ MySQL found${NC}\n"
fi

# Create virtual environment
echo -e "${YELLOW}[3/5] Creating virtual environment...${NC}"
if [ -d ".venv" ]; then
    echo -e "${YELLOW}Virtual environment already exists. Skipping creation.${NC}"
else
    python3 -m venv .venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}[4/5] Installing dependencies...${NC}"
source .venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install flask werkzeug opencv-python numpy pymysql pillow ultralytics supervision > /dev/null 2>&1
echo -e "${GREEN}✓ Dependencies installed${NC}\n"

# Create necessary directories
echo -e "${YELLOW}[5/5] Creating required directories...${NC}"
mkdir -p alerts
mkdir -p "photo samples"
echo -e "${GREEN}✓ Directories created${NC}\n"

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${GREEN}================================${NC}\n"

echo -e "${YELLOW}Next steps:${NC}"
echo -e "1. Edit app.py and update:"
echo -e "   - Database credentials (DB_CONFIG)"
echo -e "   - Email settings (EMAIL_FROM, EMAIL_PASS)"
echo -e "   - Username/Password (USERNAME, PASSWORD)"
echo -e ""
echo -e "2. Ensure you have MySQL running"
echo -e ""
echo -e "3. Activate virtual environment:"
echo -e "   ${GREEN}source .venv/bin/activate${NC}"
echo -e ""
echo -e "4. Run the application:"
echo -e "   ${GREEN}python app.py${NC}"
echo -e ""
echo -e "5. Open in browser:"
echo -e "   ${GREEN}http://localhost:5000${NC}"
