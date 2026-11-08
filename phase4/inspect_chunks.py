from pathlib import Path

from sentence_transformers import SentenceTransformer

from chunking import chunk_by_paragraphs, chunk_by_words, split_to_fit
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
limit = model.max_seq_length

def count_tokens(text):
    return len(model.tokenizer.tokenize(text))

strategies = {
    "words (60, overlap 15)": lambda text: chunk_by_words(text, 60, 15),
    "paragraphs (max 80)": lambda text: chunk_by_paragraphs(text, 80),
    "recursive (max 120 tokens)": lambda text: split_to_fit(text, count_tokens, 120),
}

for doc_path in sorted(Path("phase4/docs").glob("*.txt")):
    text = doc_path.read_text(encoding="utf-8")
    print(f"\n######## {doc_path.name} ({len(text.split())} words) ########")

    for name, chunker in strategies.items():
        chunks = chunker(text)
        token_counts = [len(model.tokenizer.tokenize(c)) for c in chunks]
        too_long = sum(1 for t in token_counts if t > limit)

        print(f"\n--- {name}: {len(chunks)} chunks, tokens per chunk: {token_counts}")
        if too_long:
            print(f"    WARNING: {too_long} chunk(s) exceed the {limit}-token limit and will be truncated")
        print(f"    First chunk ends with: ...{chunks[0][-80:]!r}")