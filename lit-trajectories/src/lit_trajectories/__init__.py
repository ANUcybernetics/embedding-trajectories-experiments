from sentence_transformers import SentenceTransformer


def embed(model, chunks):
    return model.encode(chunks)


def main() -> int:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print(embed(model, ["one", "two"]))
    return 0
