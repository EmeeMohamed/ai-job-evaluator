import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAMES = [
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
]

sentences = [
    "Built REST APIs with FastAPI to serve sales forecasts",
    "Developed web services that expose HTTP endpoints",
    "Fluent in German, native Arabic speaker",
    "Trained a classification model with scikit-learn",
    "Built a machine learning model to predict customer churn",
    "Organized events and presented at information booths",
    "Erfahrung mit der Entwicklung von Webanwendungen",
]

queries = [
    "Experience with backend web development",
    "Deutschkenntnisse",
    "Sprachkenntnisse",
    "machine learning experience",
]

for model_name in MODEL_NAMES:
    model = SentenceTransformer(model_name)
    doc_vectors = model.encode(sentences, normalize_embeddings=True)

    print(f"\n===== {model_name} =====")
    for query in queries:
        query_vector = model.encode(query, normalize_embeddings=True)
        scores = doc_vectors @ query_vector
        ranking = np.argsort(scores)[::-1]

        print(f"\nQuery: {query}")
        for i in ranking[:2]:
            print(f"  {scores[i]:.3f}  {sentences[i]}")