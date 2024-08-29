import lit_trajectories.chunker as chunker


def aesop_fables():
    with open("data/aesop.txt", "r") as file:
        content = file.read()
    fables = content.split("\n\n\n\n")
    unwrapped_fables = []
    for fable in fables:
        paragraphs = fable.split("\n\n")
        title = paragraphs[0].strip()
        unwrapped_paragraphs = [" ".join(para.split("\n")) for para in paragraphs[1:]]
        unwrapped_fable = "\n\n".join(unwrapped_paragraphs)
        unwrapped_fables.append((title, unwrapped_fable.strip()))
    return [(title, 0, fable) for title, fable in unwrapped_fables if fable]


def aesop_paragraphs():
    fables = aesop_fables()
    paragraphs = []
    for title, _, fable in fables:
        fable_paragraphs = [p.strip() for p in fable.split("\n\n") if p.strip()]
        for index, paragraph in enumerate(fable_paragraphs):
            paragraphs.append((title, index, paragraph))
    return paragraphs


def aesop_sentences():
    fables = aesop_fables()
    sentences = []
    for title, _, fable in fables:
        fable_sentences = chunker.sentences(fable)
        for index, sentence in enumerate(fable_sentences):
            if sentence.strip():
                sentences.append((title, index, sentence.strip()))
    return sentences
