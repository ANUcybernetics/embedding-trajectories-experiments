from sentence_transformers import SentenceTransformer

# global variable... gross
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(chunks):
    return embed_model.encode(chunks)
