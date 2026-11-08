import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

sentences = [
    "Built REST APIs with FastAPI to serve sales forecasts",
    "Developed web services that expose HTTP endpoints",
    "Fluent in German, native Arabic speaker",
    "Trained a classification model with scikit-learn",
    "Built a machine learning model to predict customer churn",
    "Organized events and presented at information booths",
]

embeddings = model.encode(sentences,normalize_embeddings=True)

print("Shape:", embeddings.shape)
print("First 8 numbers of sentence 1:", np.round(embeddings[0][:8], 3))

query = "Experience with backend web development"
query_embedding = model.encode(query, normalize_embeddings=True)

scores = embeddings @ query_embedding

print(f"\nQuery: {query}\n")
for score, sentence in sorted(zip(scores, sentences), reverse=True):
    print(f"{score:.3f}  {sentence}")