from search import model, search

query = "What is my master's thesis about?"
hit = search(query)[0]

chunk_vector = model.encode(hit["text"], normalize_embeddings=True)
query_vector = model.encode(query, normalize_embeddings=True)
direct = float(chunk_vector @ query_vector)

print(f"From Chroma (converted): {hit['similarity']:.4f}")
print(f"Calculated directly:     {direct:.4f}")