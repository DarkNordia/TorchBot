from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    import torch
except ModuleNotFoundError:
    print("PyTorch is not installed yet. From the project folder, run:")
    print("  .venv\\Scripts\\activate")
    print("  python -m pip install -r requirements.txt")
    print("  python src\\chat.py")
    print("Or just run:  .\\run.bat")
    sys.exit(1)

from dialogues import find_best_reply, load_pairs, looks_like_garbage
from model import ChatLSTM
from vocab import ASSISTANT, USER, Vocab, tokenize

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "chatbot.pt"
VOCAB_PATH = ROOT / "models" / "vocab.json"
DATA_PATH = ROOT / "data" / "dialogues.txt"
TRAIN_SCRIPT = ROOT / "src" / "train.py"


def ensure_model(train_if_missing: bool = True) -> None:
    if MODEL_PATH.exists() and VOCAB_PATH.exists():
        return
    if not train_if_missing:
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run: python src/train.py"
        )
    print("No trained model found. Training now (about 1–2 minutes on CPU)...")
    subprocess.check_call([sys.executable, str(TRAIN_SCRIPT)], cwd=ROOT)


def load_model(device: torch.device) -> tuple[ChatLSTM, Vocab]:
    ensure_model()
    vocab = Vocab.load(VOCAB_PATH)
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location=device)
    cfg = ckpt["config"]
    model = ChatLSTM(
        cfg["vocab_size"],
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, vocab


def build_prompt(history: list[tuple[str, str]], user_message: str) -> list[int]:
    parts: list[str] = []
    for role, text in history[-4:]:
        marker = USER if role == "user" else ASSISTANT
        parts.append(f"{marker} {text.lower().strip()}")
    parts.append(f"{USER} {user_message.lower().strip()}")
    parts.append(ASSISTANT)
    prompt = "\n".join(parts)
    return tokenize(prompt)


def trim_reply(text: str, max_words: int = 22) -> str:
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    return text.strip()


def reply(
    model: ChatLSTM,
    vocab: Vocab,
    pairs: list[tuple[str, str]],
    history: list[tuple[str, str]],
    user_message: str,
    *,
    device: torch.device,
    max_new_tokens: int = 18,
    temperature: float = 0.55,
) -> str:
    matched = find_best_reply(user_message, pairs)
    if matched:
        return matched

    # Tiny models ramble on very short or unknown inputs — avoid neural generation
    if len(tokenize(user_message)) <= 2:
        return "got it try a full sentence or ask about code study hobbies or the outdoors"

    prompt_tokens = build_prompt(history, user_message)
    prompt_ids = vocab.encode(prompt_tokens)
    stop_ids = {vocab.stoi[USER]}
    if ASSISTANT in vocab.stoi:
        stop_ids.add(vocab.stoi[ASSISTANT])

    new_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=20,
        stop_token_ids=stop_ids,
        greedy=True,
        device=device,
    )
    text = trim_reply(vocab.decode(new_ids))
    if looks_like_garbage(text):
        return (
            "i am not sure about that yet try a shorter question "
            "or add it to data dialogues.txt and run train.py"
        )
    return text or "i am not sure yet try rephrasing that"


def run_chat(device_name: str | None = None) -> None:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    print("Loading local PyTorch chatbot...")
    model, vocab = load_model(device)
    pairs = load_pairs(DATA_PATH)
    print(f"Ready on {device}. Type 'quit' to exit.\n")

    history: list[tuple[str, str]] = []
    while True:
        try:
            user_message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_message:
            continue
        if user_message.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        answer = reply(model, vocab, pairs, history, user_message, device=device)
        history.append(("user", user_message))
        history.append(("assistant", answer))
        print(f"Bot: {answer}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the local PyTorch bot.")
    parser.add_argument("--device", type=str, default=None, help="cpu or cuda")
    parser.add_argument("--no-auto-train", action="store_true")
    args = parser.parse_args()
    if args.no_auto_train:
        ensure_model(train_if_missing=False)
    run_chat(device_name=args.device)


if __name__ == "__main__":
    main()
