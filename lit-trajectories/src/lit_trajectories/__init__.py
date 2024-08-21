from sentence_transformers import SentenceTransformer


def aesop_fables():
    with open("data/aesop.txt", "r") as file:
        content = file.read()
    fables = content.split("\n\n\n\n")
    return [fable.strip() for fable in fables if fable.strip()]


def aesop_sentences():
    with open("data/aesop.txt", "r") as file:
        content = file.read()
    sentences = [
        " ".join(sentence.split())
        for sentence in content.replace("\n", " ").split(".")
        if sentence.strip()
    ]
    return sentences


def embed(model, chunks):
    return model.encode(chunks)


# add a download_fables function which downloads the full text of Aesop's fables from https://www.gutenberg.org/cache/epub/49010/pg49010.txt, then extracts only the text of the fables themselves (i.e. all text between the '*** START OF THE PROJECT GUTENBERG EBOOK ÆSOP'S FABLES: A VERSION FOR YOUNG READERS ***' lines)


def main() -> int:
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    return 0
