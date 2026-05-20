from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from vocab import ASSISTANT, USER, tokenize

# Map loose user intent -> exact training user line in dialogues.txt
INTENT_TO_USER_LINE: list[tuple[set[str], str]] = [
    ({"outdoor", "outdoors", "outside", "hiking", "camping"}, "what can i do outdoors"),
    ({"time", "clock", "hour"}, "what time is it"),
    ({"weather", "rain", "sunny", "temperature"}, "what is the weather"),
    ({"thank", "thanks", "appreciate"}, "thank you"),
    ({"bye", "goodbye", "later"}, "goodbye"),
    ({"joke", "funny", "laugh"}, "tell me a joke"),
    ({"pytorch"}, "what is pytorch"),
    ({"python"}, "what is python"),
    ({"retrain", "training"}, "how do i train you"),
    ({"help"}, "help"),
    ({"privacy", "local", "offline"}, "local only"),
    ({"sad", "lonely", "anxious"}, "i am sad"),
    ({"tired", "exhausted"}, "i am tired"),
    ({"stress", "overwhelmed"}, "i am stressed"),
    ({"bored"}, "i am bored"),
]

SHORT_REPLIES: dict[str, str] = {
    "ok": "sounds good what would you like to talk about next",
    "okay": "sounds good what would you like to talk about next",
    "k": "sounds good what would you like to talk about next",
    "yes": "great what should we talk about",
    "yeah": "great what should we talk about",
    "yep": "great what should we talk about",
    "no": "alright we can switch topics whenever you want",
    "nope": "alright we can switch topics whenever you want",
    "cool": "glad you think so what is next",
    "nice": "happy to hear that",
    "hmm": "take your time what is on your mind",
    "hm": "take your time what is on your mind",
    "idk": "no problem we can keep it simple what sounds interesting",
    "sure": "sounds good go ahead",
    "thanks": "you are welcome anytime",
    "ty": "you are welcome anytime",
}


def load_pairs(path: Path) -> list[tuple[str, str]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    pairs: list[tuple[str, str]] = []
    i = 0
    while i + 1 < len(lines):
        user_line, asst_line = lines[i], lines[i + 1]
        if user_line.startswith(USER) and asst_line.startswith(ASSISTANT):
            pairs.append(
                (
                    user_line[len(USER) :].strip(),
                    asst_line[len(ASSISTANT) :].strip(),
                )
            )
            i += 2
        else:
            i += 1
    return pairs


def _pair_lookup(pairs: list[tuple[str, str]], user_line: str) -> str | None:
    target = user_line.lower().strip()
    for user_text, assistant_text in pairs:
        if user_text.lower().strip() == target:
            return assistant_text
    return None


def _keyword_reply(query: str, pairs: list[tuple[str, str]]) -> str | None:
    q_tokens = set(tokenize(query))
    for keys, user_line in INTENT_TO_USER_LINE:
        if keys & q_tokens:
            hit = _pair_lookup(pairs, user_line)
            if hit:
                return hit
    return None


def _query_variants(message: str) -> list[str]:
    base = message.lower().strip()
    variants = [base]
    for part in re.split(r"[,.!?;]+", base):
        part = part.strip()
        if len(part) >= 4 and part not in variants:
            variants.append(part)
    return variants


def _score(query: str, candidate: str) -> float:
    if query == candidate:
        return 2.0

    seq = SequenceMatcher(None, query, candidate).ratio()
    q_tokens = set(tokenize(query))
    c_tokens = set(tokenize(candidate))
    if not q_tokens or not c_tokens:
        return seq

    inter = q_tokens & c_tokens
    if not inter:
        return seq * 0.25

    precision = len(inter) / len(q_tokens)
    recall = len(inter) / len(c_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    combined = 0.55 * seq + 0.45 * f1

    extra_in_query = q_tokens - c_tokens
    if extra_in_query:
        combined *= max(0.15, 1.0 - 0.75 * len(extra_in_query) / len(q_tokens))

    # "what can i do" vs "what can you do"
    if "i" in q_tokens and "you" in c_tokens and "you" not in q_tokens:
        combined *= 0.25
    if "you" in q_tokens and "i" in c_tokens and "i" not in q_tokens:
        combined *= 0.25

    if len(q_tokens) > len(c_tokens) + 2 and precision < 0.5:
        combined *= 0.55
    if query in candidate or candidate in query:
        combined += 0.35
    return combined


def find_best_reply(user_message: str, pairs: list[tuple[str, str]]) -> str | None:
    query = user_message.lower().strip()
    if not query:
        return None

    if query in SHORT_REPLIES:
        return SHORT_REPLIES[query]

    keyword = _keyword_reply(query, pairs)
    if keyword:
        return keyword

    best_reply: str | None = None
    best_score = 0.0
    best_precision = 0.0

    for variant in _query_variants(query):
        q_tokens = set(tokenize(variant))
        for user_text, assistant_text in pairs:
            candidate = user_text.lower().strip()
            score = _score(variant, candidate)
            if not q_tokens:
                continue
            inter = q_tokens & set(tokenize(candidate))
            precision = len(inter) / len(q_tokens)
            if score > best_score:
                best_score = score
                best_precision = precision
                best_reply = assistant_text

    if best_score >= 2.0:
        return best_reply
    if best_score >= 0.68 and best_precision >= 0.6:
        return best_reply
    return None


def looks_like_garbage(text: str) -> bool:
    if not text or len(text.split()) < 2:
        return True
    lowered = text.lower()
    garbage_markers = (
        "both good",
        "both are",
        "good night sleep",
        "how are you how are",
        "tea for calm coffee",
        "cats or dogs",
        "capital of germany",
        "never which",
        "win files",
    )
    return any(marker in lowered for marker in garbage_markers)
