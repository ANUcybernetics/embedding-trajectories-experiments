import lit_trajectories.texts as texts
import lit_trajectories.embedder as embedder
import lit_trajectories.vis as vis


def main() -> int:
    chunks = texts.aesop_paragraphs()
    embeddings = embedder.embed_as_ndarray(chunks)

    # print(embeddings[0])
    vis.calculate_trimap(embeddings)

    return 0
