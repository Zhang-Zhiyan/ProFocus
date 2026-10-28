
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import wordnet
    _NLTK_OK = True
except Exception:
    nltk = None
    WordNetLemmatizer = None
    wordnet = None
    _NLTK_OK = False


def get_wordnet_pos(treebank_tag):
    if not _NLTK_OK:
        return None
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

lemmatizer = WordNetLemmatizer() if _NLTK_OK else None

def lemmatize_sentence(sentence):
    if not _NLTK_OK:
        return [str(w).lower() for w in sentence]
    lemmatized_output = [lemmatizer.lemmatize(w, get_wordnet_pos(pos)) for w, pos in nltk.pos_tag(sentence)]
    return lemmatized_output
