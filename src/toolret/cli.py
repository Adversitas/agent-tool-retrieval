"""``toolret`` command line: prepare | finetune | benchmark | search | serve | report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import data as data_mod
from .evaluate import evaluate, prompt_budget
from .report import write_report
from .retrievers import BGE_QUERY_PREFIX, BM25, Dense, Hybrid

BASE_MODEL = "BAAI/bge-small-en-v1.5"


def _dense(tools, model_path: str, name: str) -> Dense:
    from sentence_transformers import SentenceTransformer

    return Dense(tools, SentenceTransformer(model_path, device="cpu"), BGE_QUERY_PREFIX, name)


def _retriever(kind: str, tools, model_path: str):
    bm25 = BM25(tools)
    if kind == "bm25":
        return bm25
    dense = _dense(tools, model_path, "dense")
    return dense if kind == "dense" else Hybrid(bm25, dense)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="toolret", description=__doc__)
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--reports", type=Path, default=Path("reports"))
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare", help="download Glaive v2 and build tools/queries jsonl")
    pr.add_argument("--raw", type=Path, default=Path("data/raw/glaive-function-calling-v2.json"))
    ft = sub.add_parser("finetune", help="contrastive fine-tuning on the train split")
    ft.add_argument("--out", default="models/bge-small-tools")
    ft.add_argument("--epochs", type=int, default=1)
    b = sub.add_parser("benchmark", help="evaluate retrievers on seen/unseen test splits")
    b.add_argument("--bm25-only", action="store_true", help="no model download (CI)")
    b.add_argument("--finetuned", default="models/bge-small-tools")
    s = sub.add_parser("search", help="query the registry")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=5)
    s.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    s.add_argument("--model", default=BASE_MODEL)
    sv = sub.add_parser("serve", help="run the MCP server (stdio)")
    sv.add_argument("--retriever", choices=["bm25", "dense", "hybrid"], default="hybrid")
    sv.add_argument("--model", default=BASE_MODEL)
    sub.add_parser("report", help="rebuild reports/results.md")
    a = p.parse_args(argv)

    if a.cmd == "prepare":
        raw = data_mod.download(a.raw)
        samples = json.loads(raw.read_text(encoding="utf-8"))
        tools, queries = data_mod.build(samples)
        data_mod.save(tools, queries, a.data)
        print(f"{len(tools)} tools, {len(queries)} queries -> {a.data}")
        return

    tools, queries = data_mod.load(a.data)
    if a.cmd == "finetune":
        from .finetune import finetune

        train = [q for q in queries if q.split == "train"]
        finetune(BASE_MODEL, tools, train, a.out, epochs=a.epochs)
    elif a.cmd == "benchmark":
        retrievers = [BM25(tools)]
        if not a.bm25_only:
            base = _dense(tools, BASE_MODEL, "bge-small (base)")
            retrievers += [base, Hybrid(retrievers[0], base, "hybrid BM25 + bge-small")]
            if Path(a.finetuned).exists():
                tuned = _dense(tools, a.finetuned, "bge-small (fine-tuned)")
                retrievers += [tuned, Hybrid(retrievers[0], tuned, "hybrid BM25 + fine-tuned")]
        results = {"prompt_budget": prompt_budget(tools), "runs": []}
        for r in retrievers:
            for split in ("test_seen", "test_unseen"):
                qs = [q for q in queries if q.split == split]
                res = evaluate(r, tools, qs)
                res.update(retriever=r.name, split=split)
                results["runs"].append(res)
                print(
                    f"{r.name:28s} {split:12s} R@1={res['recall@1']:.3f} "
                    f"R@5={res['recall@5']:.3f} MRR={res['mrr']:.3f}"
                )
        a.reports.mkdir(parents=True, exist_ok=True)
        name = "benchmark_bm25.json" if a.bm25_only else "benchmark.json"
        (a.reports / name).write_text(json.dumps(results, indent=1))
    elif a.cmd == "search":
        from .server import Registry

        reg = Registry(tools, _retriever(a.retriever, tools, a.model))
        for hit in reg.search(a.query, a.k):
            print(f"{hit['score']:.4f}  {hit['name']}: {hit['description']}")
    elif a.cmd == "serve":
        from .server import Registry, build_server

        reg = Registry(tools, _retriever(a.retriever, tools, a.model))
        build_server(reg).run("stdio")
    elif a.cmd == "report":
        print(write_report(a.reports))


if __name__ == "__main__":
    main()
