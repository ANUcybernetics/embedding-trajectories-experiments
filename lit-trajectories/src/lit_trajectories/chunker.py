from spacy.lang.en import English


def sentences(text):
    nlp = English()
    nlp.add_pipe("sentencizer")
    doc = nlp(text)
    return [sent.text for sent in doc.sents]
