import lit_trajectories.texts as texts
import lit_trajectories.embedder as embedder
import lit_trajectories.vis as vis


def main() -> int:
    chunks = texts.aesop_sentences()
    print(chunks[0:3])
    # embeddings = embedder.embed_as_ndarray(chunks)

    # vis.calculate_trimap(embeddings)

    return 0
