"""Build a tool library and a tool-retrieval benchmark from Glaive function-calling v2.

Each Glaive sample is a system prompt listing 0-2 JSON tool schemas, followed by a chat.
We keep the first user turn as the *query* and read the tool the assistant actually called
in its first reply (``<functioncall> {"name": ...}``) as the *gold tool*. Chats whose first
reply calls no tool are dropped: "no tool fits" was judged against the 1-2 tools of that
sample, not against the whole library, so it is not a valid label here.

The library pools every distinct tool name seen in any system prompt (keeping the most
frequent definition of each), which is the realistic setting for an agent platform: the agent
does not get the two tools that happen to fit, it has to find them among hundreds.

Splits are made **by tool name**, so the "unseen" test set measures what happens when a new
tool is registered after the retriever was fine-tuned: no retraining should be needed.
"""

from __future__ import annotations

import json
import random
import re
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2/resolve/main/"
    "glaive-function-calling-v2.json"
)
_DECODER = json.JSONDecoder()
_CALL = re.compile(r"<functioncall>\s*\{.*?\"name\"\s*:\s*\"([^\"]+)\"", re.S)
_FIRST_TURN = re.compile(r"USER:(.*?)ASSISTANT:(.*?)(?=USER:|FUNCTION RESPONSE:|$)", re.S)


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict

    def text(self) -> str:
        """Natural-language rendering used by every retriever."""
        props = self.parameters.get("properties", {}) if isinstance(self.parameters, dict) else {}
        params = "; ".join(
            f"{k.replace('_', ' ')}: {v.get('description', '')}" if isinstance(v, dict) else k
            for k, v in props.items()
        )
        return f"{self.name.replace('_', ' ')}. {self.description}. Parameters: {params}"

    def schema_chars(self) -> int:
        return len(json.dumps(asdict(self)))


@dataclass
class Query:
    qid: int
    text: str
    gold: str  # name of the tool the assistant called
    split: str  # train | test_seen | test_unseen


def extract_tools(system: str) -> list[dict]:
    out, i = [], system.find("{")
    while i != -1:
        try:
            obj, j = _DECODER.raw_decode(system, i)
        except json.JSONDecodeError:
            i = system.find("{", i + 1)
            continue
        if isinstance(obj, dict) and "name" in obj:
            out.append(obj)
        i = system.find("{", j)
    return out


def parse_sample(sample: dict) -> tuple[list[dict], str | None, str | None]:
    """Return (tools, first user message, called tool name or None)."""
    tools = extract_tools(sample.get("system", ""))
    m = _FIRST_TURN.search(sample.get("chat", ""))
    if not tools or not m:
        return tools, None, None
    user = " ".join(m.group(1).split())
    call = _CALL.search(m.group(2))
    return tools, user, call.group(1) if call else None


def download(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        urllib.request.urlretrieve(URL, path)
    return path


def build(
    samples: list[dict],
    seed: int = 0,
    unseen_share: float = 0.3,
    train_cap: int = 40,
    test_cap: int = 8,
) -> tuple[list[Tool], list[Query]]:
    defs: dict[str, Counter] = defaultdict(Counter)
    by_tool: dict[str, list[str]] = defaultdict(list)
    for s in samples:
        tools, user, called = parse_sample(s)
        for t in tools:
            key = json.dumps(
                {"description": t.get("description", ""), "parameters": t.get("parameters", {})},
                sort_keys=True,
            )
            defs[t["name"]][key] += 1
        if user is None or len(user) < 8:
            continue
        if called is not None:
            by_tool[called].append(user)

    library = []
    for name, counter in sorted(defs.items()):
        best = json.loads(counter.most_common(1)[0][0])
        library.append(Tool(name, best["description"], best["parameters"]))
    names = {t.name for t in library}

    rng = random.Random(seed)
    called = sorted(n for n in by_tool if n in names)
    rng.shuffle(called)
    n_unseen = int(len(called) * unseen_share)
    unseen, seen = set(called[:n_unseen]), set(called[n_unseen:])

    queries: list[Query] = []

    def add(text, gold, split):
        queries.append(Query(len(queries), text, gold, split))

    for name in sorted(by_tool):
        qs = sorted(set(by_tool[name]))
        rng.shuffle(qs)
        if name in unseen:
            for q in qs[:test_cap]:
                add(q, name, "test_unseen")
        elif name in seen:
            for q in qs[:test_cap]:
                add(q, name, "test_seen")
            for q in qs[test_cap : test_cap + train_cap]:
                add(q, name, "train")
    return library, queries


def save(library: list[Tool], queries: list[Query], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "tools.jsonl", "w", encoding="utf-8") as f:
        for t in library:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
    with open(out_dir / "queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(asdict(q), ensure_ascii=False) + "\n")


def load(data_dir: Path) -> tuple[list[Tool], list[Query]]:
    with open(data_dir / "tools.jsonl", encoding="utf-8") as f:
        library = [Tool(**json.loads(line)) for line in f]
    with open(data_dir / "queries.jsonl", encoding="utf-8") as f:
        queries = [Query(**json.loads(line)) for line in f]
    return library, queries
