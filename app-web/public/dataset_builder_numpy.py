import random
import pymorphy3
import numpy as np
import math

from params import NO_PUNCT
from navec import Navec
import itertools
from mosestokenizer import MosesPunctuationNormalizer
import urllib
import os

punctuation_normalizer = MosesPunctuationNormalizer('ru')

morph = pymorphy3.MorphAnalyzer()

def download_file(file, url):
    file_path = os.path.join("./", file)
    if os.path.isfile(file_path):
        return file_path
    print("donwloading", file, "to", file_path)
    # url = BASE_URL + file
    urllib.request.urlretrieve(url, file_path)
    return file_path

navec_path = download_file('hudlit_12B_500K_300d_100q.tar',
        "https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar")
navec = Navec.load(navec_path)


NUMPY_DTYPE = float
NAVEC_UNK = navec['<unk>']
NAVEC_UNK_TORCH = NAVEC_UNK
NAVEC_PAD_TORCH = navec['<pad>']

UNDEF_TOKEN = "UNDEF"
PAD_TOKEN = "PAD"


def empty_word_features(params):
    return np.zeros([params["TOTAL_WORD_FEATURES_CNT"]],
                        dtype=NUMPY_DTYPE)

def get_navec_start_idx(params):
    return params['VARIANT_FEATURES_CNT'] * params['VARIANTS_CNT']

def pad_word_features(params):
    res = empty_word_features(params)
    res[get_navec_start_idx(params): ] = NAVEC_PAD_TORCH
    return res

def undef_word_features(params):
    res = empty_word_features(params)
    res[get_navec_start_idx(params): ] = NAVEC_UNK_TORCH
    return res


PNCT_TAGS = {
    '.': 'PUNCT_DOT',
    '!': 'PUNCT_DOT',
    '?': 'PUNCT_DOT',
    ',': 'PUNCT_COMMA',
    '-': 'PUNCT_DASH',
    '.':'PUNCT_DOT',
    '"': 'PUNCT_QUOTE',
    '\'': 'PUNCT_QUOTE',
    '(': 'PUNCT_LEFT_PARENTHESIS',
    ')': 'PUNCT_RIGHT_PARENTHESIS',
}

def get_word_features(word, params):
    if word == PAD_TOKEN:
        return pad_word_features(params)
    if word == UNDEF_TOKEN:
        return undef_word_features(params)

    additional_tags = []

    res = empty_word_features(params)
    if not str.isalpha(word[0]):
        word_punct = punctuation_normalizer(word).strip()
        if word_punct in PNCT_TAGS:
            additional_tags.append(PNCT_TAGS[word_punct])

    if str.isupper(word[0]):
        additional_tags.append('CAPITALIZED')

    use_navec = True

    variant_features_cnt = params['VARIANT_FEATURES_CNT']
    for i, variant in enumerate(morph.parse(word)[:params["VARIANTS_CNT"]]):
        tags = variant.tag._grammemes_tuple

        for tag in itertools.chain(tags, additional_tags):
            tag_index = params["feature_tags_dict"].get(tag, None)
            if tag_index:
                res[i * variant_features_cnt + tag_index] = True
            if i == 0 and tag in params['CUT_NAVEC_TAGS_SET']:
                use_navec = False
        res[i * variant_features_cnt + params["VARIANT_PROB_IDX"]] = variant.score


    if params['USE_NAVEC'] and use_navec:
        res[get_navec_start_idx(params): ] = navec.get(word.lower(), NAVEC_UNK)

    return res

def calculate_word_features_for_tokens(input, params):
    input = [get_word_features(s, params) for s in input]
    return np.stack(input)
