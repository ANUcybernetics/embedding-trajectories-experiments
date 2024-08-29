from sentence_transformers import SentenceTransformer
import numpy as np

# global variable... gross
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(chunks):
    return embed_model.encode(chunks)


def embed_as_ndarray(chunks):
    embeddings = embed(chunks)
    return np.array(embeddings)
