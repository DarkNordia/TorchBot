from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import ChatLSTM
from vocab import Vocab, tokenize

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "dialogues.txt"
MODEL_DIR = ROOT / "models"


class DialogueDataset(Dataset):
    def __init__(self, sequences: list[list[int]], seq_len: int, pad_id: int = 0):
        window = seq_len + 1
        self.samples: list[torch.Tensor] = []
        for seq in sequences:
            if len(seq) < 2:
                continue
            if len(seq) <= window:
                chunk = seq + [pad_id] * (window - len(seq))
                self.samples.append(torch.tensor(chunk[:window], dtype=torch.long))
                continue
            for start in range(0, len(seq) - 1, seq_len):
                chunk = seq[start : start + window]
                if len(chunk) < 2:
                    continue
                if len(chunk) < window:
                    chunk = chunk + [pad_id] * (window - len(chunk))
                self.samples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


def load_dialogues(path: Path) -> list[str]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [ln for ln in lines if ln]


def build_sequences(texts: list[str], vocab: Vocab) -> list[list[int]]:
    """One sequence per user/assistant pair (avoids rambling into the next example)."""
    sequences: list[list[int]] = []
    i = 0
    while i + 1 < len(texts):
        user_line, asst_line = texts[i], texts[i + 1]
        if user_line.startswith("<|user|>") and asst_line.startswith("<|assistant|>"):
            seq = vocab.encode(tokenize(user_line)) + vocab.encode(tokenize(asst_line))
            if len(seq) >= 2:
                sequences.append(seq)
            i += 2
        else:
            i += 1
    return sequences


def train(
    epochs: int = 120,
    batch_size: int = 32,
    seq_len: int = 48,
    lr: float = 2e-3,
    device: str | None = None,
) -> None:
    device_t = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    texts = load_dialogues(DATA_PATH)
    vocab = Vocab.build(texts)
    sequences = build_sequences(texts, vocab)
    dataset = DialogueDataset(sequences, seq_len=seq_len, pad_id=vocab.pad_id)
    if len(dataset) == 0:
        raise RuntimeError("No training samples found. Check data/dialogues.txt")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model = ChatLSTM(vocab.size).to(device_t)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_id)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        steps = 0
        for batch in loader:
            batch = batch.to(device_t)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]
            logits, _ = model(inputs)
            loss = loss_fn(logits.reshape(-1, vocab.size), targets.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            steps += 1

        avg = total_loss / max(steps, 1)
        if epoch == 1 or epoch % 20 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs}  loss={avg:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "vocab_size": vocab.size,
                "embed_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
            },
        },
        MODEL_DIR / "chatbot.pt",
    )
    vocab.save(MODEL_DIR / "vocab.json")
    print(f"saved model to {MODEL_DIR / 'chatbot.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the local PyTorch chatbot.")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        device=args.device,
    )


if __name__ == "__main__":
    main()
