# TorchBot
TorchBot is a small, private AI chatbot that runs entirely on your computer. It uses pure PyTorch (a 2-layer LSTM, ~2M parameters) plus phrase matching over a dialogue file you can edit. There are no cloud APIs, no API keys, and no data sent online after install. It is aimed at learning, experimentation, and offline demos—not at replacing ChatGPT.

Instructions for installation:


Requirements:

Python 3.10 or newer (3.11 or 3.12 recommended)

~2 GB free disk space (mostly for PyTorch)

2 GB RAM minimum (4 GB+ recommended)

Windows, macOS, or Linux

Internet only for the first setup (pip install). Chat works offline after that.


Optional:

NVIDIA GPU + CUDA — faster training; not required (CPU is fine)

Python packages (installed automatically via requirements.txt)


torch>=2.0.0

Not required:


No API keys

No cloud account

No Docker


Installation
1. Install Python
Windows

Download Python from https://www.python.org/downloads/
Run the installer and check “Add python.exe to PATH”
Turn off Store aliases: Settings → Apps → Advanced app settings → App execution aliases → disable python.exe and python3.exe
Open a new terminal and run:
py -3 --version

macOS

python3 --version
# If missing: brew install python


Linux (Debian/Ubuntu)

sudo apt update
sudo apt install python3 python3-venv python3-pip
python3 --version


2. Get the project

git clone https://github.com/YOUR_USERNAME/pytorch-chatbot.git
cd pytorch-chatbot

Or download the ZIP from GitHub and extract it, then cd into the folder.

3. Create a virtual environment and install dependencies
Windows (PowerShell)


cd pytorch-chatbot
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

macOS / Linux


cd pytorch-chatbot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt


4. Train the model (first time only)

python src/train.py



How to use the bot
Start chatting
Make sure the virtual environment is activated, then:

python src/chat.py
You should see:

Loading local PyTorch chatbot...
Ready on cpu. Type 'quit' to exit.
You:
Type a message and press Enter
Type quit, exit, or q to leave
Press Ctrl+C to stop immediately
Example

You: hello
Bot: hi there how can i help you today
You: what is pytorch
Bot: pytorch is a python library for building and training neural networks with tensors
You: quit
Optional commands
python src/chat.py --device cuda      # use GPU if available
python src/train.py --epochs 200      # longer training
Customize replies
Edit data/dialogues.txt — add pairs in this format:
<|user|> your question here
<|assistant|> the answer you want
Retrain:
python src/train.py
Chat again:
python src/chat.py

Tips
Use short, clear questions for best results
Add several phrasings for the same topic in dialogues.txt
This is a small local model — not comparable to ChatGPT
Your messages stay on your machine; nothing is sent to a cloud API

Troubleshooting
Problem	Fix
Python was not found
Install Python from python.org; disable Windows Store python aliases; reopen terminal
pip not recognized
Use python -m pip install -r requirements.txt
No module named 'torch'
Activate .venv, then run python -m pip install -r requirements.txt
Wrong or random answers
Add the question/answer to dialogues.txt and run python src/train.py again
PowerShell blocks run.bat
Use .\run.bat from the project folder, or follow manual install steps above











