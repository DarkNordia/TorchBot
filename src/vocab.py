from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD = "<pad>"
UNK = "<unk>"
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
SPECIALS = [PAD, UNK, USER, ASSISTANT]


def tokenize(text: str) -> list[str]:
    text = text.lower().strip()
    if not text:
        return []
    for special in (USER, ASSISTANT):
        text = text.replace(special, f" {special} ")
    tokens: list[str] = []
    for piece in text.split():
        if piece in {USER, ASSISTANT}:
            tokens.append(piece)
        else:
            tokens.extend(re.findall(r"[a-z0-9']+", piece))
    return tokens


class Vocab:
    def __init__(self, stoi: dict[str, int], itos: list[str]):
        self.stoi = stoi
        self.itos = itos

    @property
    def size(self) -> int:
        return len(self.itos)

    @property
    def pad_id(self) -> int:
        return self.stoi["<pad>"]

    @property
    def unk_id(self) -> int:
        return self.stoi["<unk>"]

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]

    def decode(self, ids: list[int]) -> str:
        words: list[str] = []
        for i in ids:
            if i < 0 or i >= len(self.itos):
                continue
            tok = self.itos[i]
            if tok in {PAD, USER, ASSISTANT}:
                continue
            words.append(tok)
        return " ".join(words)

    @classmethod
    def build(cls, texts: list[str], max_size: int = 4000) -> Vocab:
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(tokenize(text))

        itos = list(SPECIALS)
        for word, _ in counter.most_common(max_size - len(SPECIALS)):
            if word not in itos:
                itos.append(word)

        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi, itos)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}, ensure_ascii=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Vocab:
        data = json.loads(path.read_text(encoding="utf-8"))
        itos = data["itos"]
        stoi = {tok: i for i, tok in enumerate(itos)}
        return cls(stoi, itos)
