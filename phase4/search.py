import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
DB_DIR = "phase4/chroma_db"
COLLECTION_NAME = "knowledge_base"
TOP_K = 3

model = SentenceTransformer(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=DB_DIR)
collection = client.get_collection(COLLECTION_NAME)

stored_model = collection.metadata.get("embedding_model")
if stored_model != EMBEDDING_MODEL:
    raise SystemExit(
        f"The index was built with '{stored_model}', but this script uses "
        f"'{EMBEDDING_MODEL}'. Rebuild the index with build_index.py."
    )


def search(query, k=TOP_K):
    query_vector = model.encode(query, normalize_embeddings=True)
    result = collection.query(query_embeddings=[query_vector.tolist()], n_results=k)

    hits = []
    for text, meta, distance in zip(
        result["documents"][0], result["metadatas"][0], result["distances"][0]
    ):
        hits.append({
            "text": text,
            "source": meta["source"],
            "chunk_index": meta["chunk_index"],
            "similarity": 1 - distance / 2,
        })
    return hits


def main():
    print(f"Searching {collection.count()} chunks. Type 'quit' to exit.\n")
    while True:
        query = input("Query: ")
        if query.strip().lower() == "quit":
            break
        if not query.strip():
            continue
        for hit in search(query):
            preview = hit["text"][:150].replace("\n", " ")
            print(f"  {hit['similarity']:.3f}  [{hit['source']} #{hit['chunk_index']}]  {preview}...")
        print()


if __name__ == "__main__":
    main()