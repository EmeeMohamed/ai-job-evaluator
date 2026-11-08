import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from chunking import split_to_fit

DOCS_DIR = Path("phase4/docs")
CASES = json.loads(Path("phase4/retrieval_eval.json").read_text(encoding="utf-8"))
K_VALUES = [1, 3, 5]

CONFIGS = [
    {"model": "paraphrase-multilingual-MiniLM-L12-v2", "max_tokens": 120},
    {"model": "paraphrase-multilingual-MiniLM-L12-v2", "max_tokens": 60},
]


def load_chunks(model, max_tokens):
    def count_tokens(text):
        return len(model.tokenizer.tokenize(text))

    chunks = []
    for doc_path in sorted(DOCS_DIR.glob("*.txt")):
        text = doc_path.read_text(encoding="utf-8")
        for i, chunk in enumerate(split_to_fit(text, count_tokens, max_tokens)):
            chunks.append({"id": f"{doc_path.stem}-{i}", "text": chunk})
    return chunks


def find_rank(case, chunks, order):
    for position, index in enumerate(order, start=1):
        if case["must_contain"] in chunks[index]["text"]:
            return position
    return None


def evaluate(config):
    model = SentenceTransformer(config["model"])
    chunks = load_chunks(model, config["max_tokens"])
    chunk_vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    ranks = []
    for case in CASES:
        query_vector = model.encode(case["question"], normalize_embeddings=True)
        order = np.argsort(chunk_vectors @ query_vector)[::-1]
        ranks.append(find_rank(case, chunks, order))
    return chunks, ranks


for config in CONFIGS:
    chunks, ranks = evaluate(config)
    n = len(CASES)

    print(f"\n===== {config['model']} | max {config['max_tokens']} tokens | {len(chunks)} chunks =====")
    for case, rank in zip(CASES, ranks):
        marker = "  " if rank is not None and rank <= 5 else "XX"
        print(f"  {marker} rank {str(rank):>4}  {case['question']}")

    for k in K_VALUES:
        hits = sum(1 for r in ranks if r is not None and r <= k)
        print(f"  Hit@{k}: {hits}/{n} = {hits / n:.2f}")

    mrr = sum(1 / r for r in ranks if r is not None) / n
    print(f"  MRR: {mrr:.3f}")