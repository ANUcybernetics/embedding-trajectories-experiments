import lit_trajectories.texts as texts
# import lit_trajectories.embed as embed


def main() -> int:
    chunks = texts.aesop_paragraphs()

    print(chunks[0:4])

    return 0
