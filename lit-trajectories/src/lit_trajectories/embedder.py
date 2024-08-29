from sentence_transformers import SentenceTransformer
import numpy as np
import trimap

# global variable... gross
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(chunks):
    return embed_model.encode(chunks)


def trimap_embeddings(chunks):
    embeddings = [embed(text) for (_, _, text) in chunks]
    return trimap.TRIMAP(distance="cosine", apply_pca=False).fit_transform(
        np.array(embeddings)
    )
