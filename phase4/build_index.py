from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from chunking import split_to_fit

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
MAX_TOKENS = 60
DOCS_DIR = Path("phase4/docs")
DB_DIR = "phase4/chroma_db"
COLLECTION_NAME = "knowledge_base"

model = SentenceTransformer(EMBEDDING_MODEL)


def count_tokens(text):
    return len(model.tokenizer.tokenize(text))


client = chromadb.PersistentClient(path=DB_DIR)

try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass  # the collection doesn't exist yet on the first run

collection = client.create_collection(
    name=COLLECTION_NAME,
    metadata={"embedding_model": EMBEDDING_MODEL, "max_tokens": MAX_TOKENS},
)

ids, documents, metadatas = [], [], []

for doc_path in sorted(DOCS_DIR.glob("*.txt")):
    text = doc_path.read_text(encoding="utf-8")
    chunks = split_to_fit(text, count_tokens, MAX_TOKENS)
    for i, chunk in enumerate(chunks):
        ids.append(f"{doc_path.stem}-{i}")
        documents.append(chunk)
        metadatas.append({"source": doc_path.name, "chunk_index": i})
    print(f"{doc_path.name}: {len(chunks)} chunks")

embeddings = model.encode(documents, normalize_embeddings=True)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings.tolist(),
    metadatas=metadatas,
)

print(f"\nStored {collection.count()} chunks in '{COLLECTION_NAME}' at {DB_DIR}")