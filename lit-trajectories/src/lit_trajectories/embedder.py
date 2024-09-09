from sentence_transformers import SentenceTransformer
import numpy as np
import trimap
import pacmap

# global variable... gross
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


def embed(chunks):
    return embed_model.encode(chunks)


def trimap_embeddings(chunks):
    embeddings = [embed(text) for (_, _, text) in chunks]
    return trimap.TRIMAP(distance="cosine", apply_pca=False).fit_transform(
        np.array(embeddings)
    )


def pacmap_embeddings(chunks):
    embeddings = [embed(text) for (_, _, text) in chunks]
    return pacmap.PaCMAP(
        n_components=2, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0
    ).fit_transform(np.array(embeddings))
