import lit_trajectories.texts as texts
import lit_trajectories.vis as vis


def main() -> int:
    chunks = texts.aesop_paragraphs()
    vis.trimap_scatterplot(chunks)

    return 0
