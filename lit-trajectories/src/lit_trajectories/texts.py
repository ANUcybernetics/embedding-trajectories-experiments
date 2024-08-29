import lit_trajectories.chunker as chunker


def aesop_fables():
    with open("data/aesop.txt", "r") as file:
        content = file.read()
    fables = content.split("\n\n\n\n")
    unwrapped_fables = []
    for fable in fables:
        paragraphs = fable.split("\n\n")
        unwrapped_paragraphs = [" ".join(para.split("\n")) for para in paragraphs]
        unwrapped_fable = "\n\n".join(unwrapped_paragraphs)
        unwrapped_fables.append(unwrapped_fable.strip())
    return [fable.strip() for fable in unwrapped_fables if fable]


def aesop_paragraphs():
    fables = aesop_fables()
    paragraphs = []
    for fable in fables:
        fable_paragraphs = [p.strip() for p in fable.split("\n\n") if p.strip()]
        paragraphs.extend(fable_paragraphs)
    return paragraphs


def aesop_sentences():
    fables = aesop_fables()
    sentences = []
    for fable in fables:
        fable_sentences = chunker.sentences(fable)
        sentences.extend(sent for sent in fable_sentences if sent.strip())
    return sentences
