import os
from functools import lru_cache
from pathlib import Path
from .get_wordBase import lemmatize_sentence
def get_VAD_dict(file_path):
    vad_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split('\t')
            if len(parts) == 4:
                word = parts[0]
                vad_values = [float(part) for part in parts[1:]]
                vad_dict[word] = vad_values
    return vad_dict

@lru_cache(maxsize=1)
def load_VAD_dict():
    default_path = Path(__file__).resolve().parent / 'NRC-VAD-Lexicon' / 'BipolarScale' / 'NRC-VAD-Lexicon.txt'
    file_path = Path(os.environ.get('NRC_VAD_LEXICON', str(default_path))).expanduser()
    if not file_path.is_file():
        raise FileNotFoundError(
            f'NRC-VAD lexicon not found: {file_path}. Obtain the original NRC-VAD '
            'Lexicon from https://saifmohammad.com/WebPages/nrc-vad.html under its '
            'terms, then place the BipolarScale/NRC-VAD-Lexicon.txt file under '
            'utils/NRC-VAD-Lexicon/ or set NRC_VAD_LEXICON to its full path.'
        )
    return get_VAD_dict(file_path)


def get_sentence_VAD(sentence):
    VAD_dict = load_VAD_dict()
    words_VAD_values=[]
    baseForm_tokens = lemmatize_sentence(sentence)
    for word in baseForm_tokens:
        if word in VAD_dict:
            words_VAD_values.append(VAD_dict[word])
        else:
            words_VAD_values.append([0.,0.,0.])
    return words_VAD_values
