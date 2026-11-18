"""Contrastive fine-tuning of a bi-encoder on (user query, called tool) pairs.

Multiple-negatives ranking loss: within a batch, every other tool acts as a negative. Batches
are built round-robin over tools, so a batch never contains the same tool twice (a duplicate
would be a false negative and teach the model to push a correct tool away).
"""

from __future__ import annotations

import random
from collections import defaultdict

from .data import Query, Tool
from .retrievers import BGE_QUERY_PREFIX


def round_robin_pairs(
    tools: list[Tool], queries: list[Query], seed: int = 0
) -> list[tuple[str, str]]:
    text = {t.name: t.text() for t in tools}
    per_tool: dict[str, list[str]] = defaultdict(list)
    for q in queries:
        per_tool[q.gold].append(q.text)
    rng = random.Random(seed)
    names = sorted(per_tool)
    for n in names:
        rng.shuffle(per_tool[n])
    pairs = []
    for i in range(max(map(len, per_tool.values()), default=0)):
        round_names = [n for n in names if i < len(per_tool[n])]
        rng.shuffle(round_names)
        pairs += [(BGE_QUERY_PREFIX + per_tool[n][i], text[n]) for n in round_names]
    return pairs


def no_duplicate_batches(pairs: list[tuple[str, str]], batch_size: int) -> list[list[int]]:
    """Greedy packing of pair indices into batches with distinct tool texts."""
    batches: list[list[int]] = []
    open_batches: list[tuple[list[int], set[str]]] = []
    for i, (_, tool_text) in enumerate(pairs):
        for idx, seen in open_batches:
            if tool_text not in seen:
                idx.append(i)
                seen.add(tool_text)
                break
        else:
            open_batches.append(([i], {tool_text}))
        full = [b for b in open_batches if len(b[0]) == batch_size]
        for b in full:
            batches.append(b[0])
            open_batches.remove(b)
    batches += [b[0] for b in open_batches if len(b[0]) > 1]
    return batches


def finetune(
    base_model: str,
    tools: list[Tool],
    train: list[Query],
    out_dir: str,
    epochs: int = 1,
    batch_size: int = 32,
    lr: float = 2e-5,
    seed: int = 0,
):
    import torch
    from sentence_transformers import SentenceTransformer, losses

    torch.manual_seed(seed)
    model = SentenceTransformer(base_model, device="cpu")
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    pairs = round_robin_pairs(tools, train, seed)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    step = 0
    for _ in range(epochs):
        for batch in no_duplicate_batches(pairs, batch_size):
            queries = [pairs[i][0] for i in batch]
            docs = [pairs[i][1] for i in batch]
            features = [model.tokenize(queries), model.tokenize(docs)]
            loss = loss_fn(features, labels=None)
            optim.zero_grad()
            loss.backward()
            optim.step()
            step += 1
            if step % 20 == 0:
                print(f"step {step} loss {loss.item():.4f}")
    model.eval()
    model.save(out_dir)
    return model
