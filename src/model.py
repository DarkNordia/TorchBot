from __future__ import annotations

import torch
import torch.nn as nn


class ChatLSTM(nn.Module):
    """Small causal language model for local chat."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb = self.dropout(self.embedding(x))
        out, hidden = self.lstm(emb, hidden)
        logits = self.head(self.dropout(out))
        return logits, hidden

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        max_new_tokens: int = 40,
        temperature: float = 0.9,
        top_k: int = 40,
        eos_id: int | None = None,
        stop_token_ids: set[int] | None = None,
        greedy: bool = False,
        device: torch.device | None = None,
    ) -> list[int]:
        self.eval()
        device = device or next(self.parameters()).device
        ids = list(prompt_ids)
        hidden = None
        stop_token_ids = stop_token_ids or set()

        for _ in range(max_new_tokens):
            x = torch.tensor([ids[-1:]], dtype=torch.long, device=device)
            logits, hidden = self(x, hidden)
            logits = logits[:, -1, :] / max(temperature, 1e-6)

            if greedy:
                next_id = int(logits.argmax(dim=-1).item())
            else:
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < values[:, [-1]]] = float("-inf")
                probs = torch.softmax(logits, dim=-1)
                next_id = int(torch.multinomial(probs, num_samples=1).item())

            if next_id in stop_token_ids:
                break
            ids.append(next_id)
            if eos_id is not None and next_id == eos_id:
                break

        return ids[len(prompt_ids) :]
