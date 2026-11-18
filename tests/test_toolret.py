import asyncio
import json

import numpy as np
import pytest

from toolret.data import Query, Tool, build, extract_tools, load, parse_sample, save
from toolret.evaluate import evaluate, gold_ranks, macro_by_tool, metrics_from_ranks
from toolret.finetune import no_duplicate_batches, round_robin_pairs
from toolret.report import write_report
from toolret.retrievers import BM25, Dense, Hybrid, rrf, tokenize
from toolret.server import Registry, build_server

TOOLS = [
    Tool("get_weather", "Get the current weather for a city", {"properties": {"city": {}}}),
    Tool("convert_currency", "Convert an amount between currencies", {"properties": {}}),
    Tool("calculate_bmi", "Calculate body mass index from weight and height", {}),
    Tool("search_recipes", "Find recipes using given ingredients", {}),
]


def sample(tool: dict, user: str, call: str | None) -> dict:
    system = "SYSTEM: You have access to -\n" + json.dumps(tool, indent=2)
    reply = f'<functioncall> {{"name": "{call}", "arguments": "{{}}"}}' if call else "Sorry."
    return {"system": system, "chat": f"USER: {user}\n\n\nASSISTANT: {reply} <|endoftext|>"}


def test_extract_and_parse_glaive_sample():
    tool = {"name": "get_weather", "description": "Weather", "parameters": {"type": "object"}}
    tools, user, called = parse_sample(sample(tool, "Weather in Paris?", "get_weather"))
    assert [t["name"] for t in tools] == ["get_weather"]
    assert user == "Weather in Paris?" and called == "get_weather"
    assert parse_sample(sample(tool, "Book a flight", None))[2] is None
    two = "A {broken json} " + json.dumps(tool) + json.dumps({**tool, "name": "x"})
    assert [t["name"] for t in extract_tools(two)] == ["get_weather", "x"]


def test_build_splits_by_tool_name(tmp_path):
    tools = [
        {"name": f"tool_{i}", "description": f"does thing {i}", "parameters": {}} for i in range(10)
    ]
    samples = [
        sample(t, f"please run tool {t['name']} variant {j}", t["name"])
        for t in tools
        for j in range(12)
    ]
    library, queries = build(samples, unseen_share=0.3, test_cap=2, train_cap=5)
    assert len(library) == 10
    unseen = {q.gold for q in queries if q.split == "test_unseen"}
    trained = {q.gold for q in queries if q.split == "train"}
    assert len(unseen) == 3 and not unseen & trained
    save(library, queries, tmp_path)
    lib2, q2 = load(tmp_path)
    assert lib2 == library and q2 == queries


def test_tokenize_and_bm25_ranking():
    # "get" starts half of all tool names, so it is a stop word for tool retrieval
    assert tokenize("Can you get_weather in Paris?") == ["weather", "paris"]
    bm25 = BM25(TOOLS)
    assert bm25.search("what's the weather in Lyon", 1)[0][0].name == "get_weather"
    assert bm25.search("convert 100 dollars to euros currencies", 1)[0][0].name == (
        "convert_currency"
    )


def test_rrf_rewards_agreement():
    a = np.array([3.0, 2.0, 1.0])
    b = np.array([1.0, 3.0, 2.0])
    fused = rrf([a, b])
    assert fused.argmax() == 1  # ranked 2nd and 1st beats 1st and 3rd


class FakeEncoder:
    """Deterministic bag-of-words embedder standing in for sentence-transformers."""

    vocab = ["weather", "currency", "bmi", "recipe", "city", "money", "ingredients"]

    def encode(self, texts, batch_size=64, normalize_embeddings=True):
        out = np.array([[t.lower().count(w) + 0.01 for w in self.vocab] for t in texts], float)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_dense_and_hybrid_with_fake_encoder():
    dense = Dense(TOOLS, FakeEncoder(), name="fake")
    assert dense.search("cook with these ingredients, recipe ideas", 1)[0][0].name == (
        "search_recipes"
    )
    hybrid = Hybrid(BM25(TOOLS), dense)
    assert hybrid.scores_batch(["weather city"]).shape == (1, len(TOOLS))


def test_ranks_and_metrics():
    scores = np.array([[0.1, 0.9, 0.5], [0.8, 0.2, 0.8]])
    tools = TOOLS[:3]
    ranks = gold_ranks(scores, tools, ["convert_currency", "get_weather"])
    assert ranks.tolist() == [1, 2]  # tie counts against the gold tool
    m = metrics_from_ranks(ranks)
    assert m["recall@1"] == 0.5 and m["mrr"] == pytest.approx(0.75)
    assert macro_by_tool(np.array([1, 1, 5]), ["a", "a", "b"]) == 0.5
    qs = [Query(0, "weather in Rome", "get_weather", "test_seen")]
    assert evaluate(BM25(TOOLS), TOOLS, qs)["recall@1"] == 1.0


def test_finetune_batches_never_repeat_a_tool():
    queries = [Query(i, f"q{i}", TOOLS[i % 3].name, "train") for i in range(30)]
    pairs = round_robin_pairs(TOOLS, queries)
    assert len(pairs) == 30
    for batch in no_duplicate_batches(pairs, batch_size=3):
        docs = [pairs[i][1] for i in batch]
        assert len(docs) == len(set(docs))


def test_registry_and_mcp_server_expose_two_tools(tmp_path):
    reg = Registry(TOOLS, BM25(TOOLS))
    assert reg.search("weather", 50)[0]["name"] == "get_weather"
    assert len(reg.search("weather", 50)) == 4  # k is capped by library size
    assert reg.schema("calculate_bmi")["description"].startswith("Calculate")
    with pytest.raises(KeyError):
        reg.schema("nope")

    server = build_server(reg)
    listed = asyncio.run(server.list_tools())
    assert sorted(t.name for t in listed) == ["get_tool_schema", "search_tools"]
    result = asyncio.run(server.call_tool("search_tools", {"query": "body mass index", "k": 1}))
    assert "calculate_bmi" in json.dumps(result.model_dump(mode="json"))


def test_report_renders(tmp_path):
    run = {
        "retriever": "bm25",
        "split": "test_seen",
        "n": 3,
        "recall@1": 0.5,
        "recall@3": 0.6,
        "recall@5": 0.7,
        "recall@10": 0.8,
        "mrr": 0.55,
        "macro_recall@1": 0.4,
        "ms_per_query": 1.0,
    }
    budget = {
        "library_tools": 4,
        "all_schemas_tokens_approx": 100,
        "top5_schemas_tokens_approx": 20,
    }
    (tmp_path / "benchmark.json").write_text(json.dumps({"prompt_budget": budget, "runs": [run]}))
    assert "| bm25 | test_seen | 3 |" in write_report(tmp_path).read_text(encoding="utf-8")
