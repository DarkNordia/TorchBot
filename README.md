# TorchBot

TorchBot is a small, private AI chatbot that runs entirely on your computer. It uses pure PyTorch with a lightweight 2-layer LSTM model (~2M parameters) and phrase matching over a customizable dialogue file.

There are:

- No cloud APIs
- No API keys
- No online data collection after installation

TorchBot is designed for:

- Learning AI fundamentals
- Experimentation
- Offline demos
- Small local chatbot projects

It is **not** intended to replace large AI systems like ChatGPT.

---

# Features

- Fully offline after setup
- Pure PyTorch implementation
- Editable training dataset
- Included pre-trained model
- CPU and GPU support
- Cross-platform support
- Simple command-line interface

---

# Requirements

## Required

- Python 3.10 or newer
  - Python 3.11 or 3.12 recommended
- PyTorch
- ~2 GB free disk space
- 2 GB RAM minimum
  - 4 GB+ recommended
- Windows, macOS, or Linux

Internet is only needed during the initial setup for installing dependencies.  
After installation, TorchBot works completely offline.

---

## Optional

- NVIDIA GPU with CUDA support
  - Faster training
  - Not required for chatting

Dependencies are installed automatically through `requirements.txt`.

---

# Not Required

- API keys
- Cloud accounts
- Docker
- Internet connection after setup

---

# Quick Start

A pre-trained model is already included in `models/`.

You can immediately start chatting with:

```bash
python src/chat.py
```

Or retrain the model after editing `dialogues.txt`:

```bash
python src/train.py
```

---

# Installation

## 1. Install Python

### Windows

Download Python:

https://www.python.org/downloads/

During installation:

- Check **"Add python.exe to PATH"**
- Disable Windows Store Python aliases:
  - Settings → Apps → Advanced app settings
  - App execution aliases
  - Disable:
    - `python.exe`
    - `python3.exe`

Verify installation:

```powershell
py -3 --version
```

---

### macOS

Check if Python is installed:

```bash
python3 --version
```

If missing:

```bash
brew install python
```

---

### Linux (Debian / Ubuntu)

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip
python3 --version
```

---

# 2. Clone the Repository

```bash
git clone https://github.com/DarkNordia/TorchBot.git
cd TorchBot
```

Or download the ZIP from GitHub and extract it manually.

---

# 3. Create a Virtual Environment

## Windows (PowerShell)

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

# 4. Train the Model

Run this once after installation or whenever you edit the training data.

```bash
python src/train.py
```

---

# Usage

## Start the Chatbot

Make sure the virtual environment is activated, then run:

```bash
python src/chat.py
```

You should see:

```text
Loading local PyTorch chatbot...
Ready on cpu. Type 'quit' to exit.
You:
```

---

## Example Conversation

```text
You: hello
Bot: hi there how can i help you today

You: what is pytorch
Bot: pytorch is a python library for building and training neural networks with tensors

You: quit
```

---

# Optional Commands

## Use GPU (CUDA)

```bash
python src/chat.py --device cuda
```

---

## Train Longer

```bash
python src/train.py --epochs 200
```

---

# Customizing Responses

Edit:

```text
data/dialogues.txt
```

Add dialogue pairs in this format:

```text
<|user|> your question here
<|assistant|> the answer you want
```

Retrain the model:

```bash
python src/train.py
```

Start chatting again:

```bash
python src/chat.py
```

---

# Tips

- Use short and clear questions for better responses
- Add multiple phrasings for the same topic
- More training data improves reply quality
- TorchBot is a small local model and is not comparable to large cloud AI systems
- All messages stay on your device

---

# Troubleshooting

| Problem | Fix |
|---|---|
| Python was not found | Install Python from python.org and disable Windows Store aliases |
| `pip` not recognized | Use `python -m pip install -r requirements.txt` |
| `No module named 'torch'` | Activate `.venv` and reinstall requirements |
| Random or incorrect answers | Add more training pairs and retrain the model |
| PowerShell blocks scripts | Run from the project folder or use manual installation steps |

---

# Project Structure

```text
TorchBot/
├── data/
│   └── dialogues.txt
├── models/
│   └── chatbot_model.pt
├── src/
│   ├── chat.py
│   └── train.py
├── requirements.txt
└── README.md
```

---

# License

This project is open source.  
Feel free to modify, learn from, and experiment with it.
