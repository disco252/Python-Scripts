HASHAUTO.py 

This is a Python script for Windows(I know, gross) designed to streamline the process of cracking multiple WiFi handshake hash files using Hashcat.

It optionally tranfers handshake files from a Raspberry Pi(pwnagotchi) via SCP, then iterates through all .2200 and hc2200 hash files in the specified directory. 

The script tracks progress, prints real-time feedback from Hashcrack, and saves the cracked passwords to an output folder. 

Large-scale, repeated hash processing more practical and efficient. You are able to use all the basic hashcat commands with this, as well.

REQUIREMENTS:

Python 3

Hashcat

A valid hashcat wordlist AND rule file. 

Working SCP.

Set HASHCAT_PATH, HASHES_DIR, WORDLIST, OUTPUT_DIR, and (optionally) the SCP source in the script.

Launch HASHAUTO.py from command prompt, via python3 HASHAUTO.py after using "cd" to get to the directory. 


=============================================================================================================================================================

discordllm.py

This python script interfaces with qwen3:14b or other LLMs to relay to discord, enabling a remote access AI through Discord. 

It avoids the 2000 character limit, provides information on what step it is on, and also reports resource usage via discord. 

It utilizes perpletxity AI, or another AI via API key, to review answers it will provide and contextual information in conjunction with the LLM running on your GPUs.

It has DuckDuckGo search integration, with SerpAPI fallback utilizing Google if DuckDuckGo cannot find anything. 

You're able to adjust token limiations given the available hardware, for different thinking, fast, search, or ask modes. 

!ask <question> - Full reasoning mode with automatic web search when needed<br>
!fast <question> - Quick responses without deep reasoning<br>
!search <question> - Force web search integration with detailed analysis<br>
!check <question> - Direct Perplexity API fact-checking and context<br>

This script also has GPU out-of-memory recovery built in, including warnings and logging output, with automatic cleanup upon CTRL+C.


Here is a .bat file to make execution a bit more easy:

@echo off
REM ──────────────────────────────────────────────────────────────────────────────
REM Batch script to activate environment and run Qwen3-14B Discord Bot
REM ──────────────────────────────────────────────────────────────────────────────

REM 1. Change to the script’s directory
cd /d "PYTHON SCRIPT"

REM 2. Activate your Python virtual environment (adjust path if needed)
IF EXIST "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "venv\Scripts\activate.bat"
) ELSE (
    echo No virtual environment found. Ensure Python and dependencies are installed globally.
)

REM 3. Set necessary environment variables
set DISCORD_TOKEN=DISCORD TOKEN

REM 4. Run the bot script
echo Starting Qwen3-14B Discord Bot...
python "PYTHON SCRIPT"

REM 5. Pause to view any error messages
pause
