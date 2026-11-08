from sentence_transformers import SentenceTransformer

model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
print("Max tokens per input:", model.max_seq_length)

key = "I hold a Goethe-Zertifikat B1 in German."
filler = "I enjoy working on software projects together with my team. " * 40
query = "German language certificate"

texts = {
    "key sentence only": key,
    "key at START of long text": key + " " + filler,
    "key at END of long text": filler + key,
    "filler only (no key)": filler,
}

query_vector = model.encode(query, normalize_embeddings=True)

print(f"\nQuery: {query}\n")
for name, text in texts.items():
    vector = model.encode(text, normalize_embeddings=True)
    score = float(vector @ query_vector)
    print(f"{score:.3f}  {name}  ({len(text.split())} words)")