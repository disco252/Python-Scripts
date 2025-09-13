@echo off
REM ──────────────────────────────────────────────────────────────────────────────
REM Enhanced DeepSeek-R1 Discord Bot with OSINT Tools (Direct Python)
REM ──────────────────────────────────────────────────────────────────────────────
cd /d "E:\deepseek"

REM Create OSINT tools directory
if not exist osint_tools mkdir osint_tools

REM Performance environment variables
set MKL_NUM_THREADS=1
set NUMBA_CACHE_DIR=E:\deepseek\.cache
set TRANSFORMERS_CACHE=E:\deepseek\.cache\transformers
set HF_HOME=E:\deepseek\.cache\huggingface

REM Create main bot virtual environment
if not exist venv (
    echo Creating main bot virtual environment...
    python -m venv venv --clear
    call venv\Scripts\activate
    python -m pip install --upgrade pip setuptools wheel
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    pip install transformers accelerate bitsandbytes discord.py duckduckgo-search requests psutil
    pip install beautifulsoup4 lxml aiofiles
    echo ✅ Main virtual environment created
) else (
    call venv\Scripts\activate
)

REM Setup Sherlock
if not exist osint_tools\sherlock (
    echo 🔍 Setting up Sherlock...
    cd osint_tools
    git clone https://github.com/sherlock-project/sherlock.git
    cd sherlock
    
    REM Create dedicated virtual environment for Sherlock
    python -m venv sherlock_env --clear
    call sherlock_env\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    REM Test installation
    echo Testing Sherlock installation...
    python sherlock.py --help
    if errorlevel 0 (
        echo ✅ Sherlock installed successfully
    ) else (
        echo ⚠️ Sherlock installation may have issues
    )
    
    deactivate
    cd ..\..
)

REM Setup SpiderFoot
if not exist osint_tools\spiderfoot (
    echo 🕷️ Setting up SpiderFoot...
    cd osint_tools
    git clone https://github.com/smicallef/spiderfoot.git
    cd spiderfoot
    pip install -r requirements.txt
    cd ..\..
    echo ✅ SpiderFoot installed
)

REM Setup TheHarvester with dedicated virtual environment
if not exist osint_tools\theHarvester (
    echo 🌾 Setting up TheHarvester...
    cd osint_tools
    git clone https://github.com/laramies/theHarvester.git
    cd theHarvester
    
    REM Create dedicated virtual environment for TheHarvester
    python -m venv harvester_env --clear
    call harvester_env\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    
    REM Test installation
    echo Testing TheHarvester installation...
    python theHarvester.py -h
    if errorlevel 0 (
        echo ✅ TheHarvester installed successfully
    ) else (
        echo ⚠️ TheHarvester installation may have issues
    )
    
    deactivate
    cd ..\..
)

REM Set Discord token
set DISCORD_TOKEN=YOUR TOKEN

REM Launch enhanced bot
echo 🚀 Starting Enhanced DeepSeek OSINT Bot with Sherlock Integration...
python bot.py
pause