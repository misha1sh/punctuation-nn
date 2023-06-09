import micropip

import jsinfer

micropip.add_mock_package("docopt", "0.6.2", modules = {
    "docopt": """
        docopt = 1
    """
})

for i in "pymorphy3 numpy navec setuptools razdel".split():
    await micropip.install(i)
# await micropip.install("")/
# await micropip.install("sacremoses")
# await micropip.install("")
# await micropip.install("")
# await micropip.install("")

import numpy as np
import random
import pymorphy3
import numpy as np
import math
import pickle
from razdel import tokenize

NO_PUNCT = 0
from navec import Navec
import itertools
# from sacremoses import MosesPunctNormalizer
from pyodide.http import pyfetch
import os

# punctuation_normalizer = MosesPunctNormalizer('ru')

morph = pymorphy3.MorphAnalyzer()

async def download_file(file, url):
    file_path = os.path.join("./", file)
    if os.path.isfile(file_path):
        return file_path
    print("donwloading", file, "to", file_path)
    # url = BASE_URL + file
    response = await pyfetch(url)
    with open(file, "wb") as f:
        f.write(await response.bytes())
    return file_path

navec_path = await download_file('hudlit_12B_500K_300d_100q.tar',
                                 "https://storage.yandexcloud.net/misha-sh-objects/hudlit_12B_500K_300d_100q.tar")
        # "/hudlit_12B_500K_300d_100q.tar")
navec = Navec.load(navec_path)

response = await pyfetch("/params.pickle")
params = pickle.loads(await response.bytes())

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
    #'\\'': 'PUNCT_QUOTE',
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
        # word_punct = punctuation_normalizer(word).strip()
        word_punct = word.strip()[0]
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


from collections import deque
import random

#https://stackoverflow.com/a/15993515
class ListRandom(object):
    def __init__(self):
        self.items = []

    def add_item(self, item):
        self.items.append(item)

    def remove_item(self, position):
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item

    def __len__(self):
        return len(self.items)

    def pop_random(self):
        assert len(self.items) > 0
        i = random.randrange(0, len(self.items))
        element = self.items[i]
        self.remove_item(i)
        return element




class Stream:
    def __init__(self, generator):
        try:
            self.generator = iter(generator)
        except TypeError:
            self.generator = generator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    @staticmethod
    def repeat(element, n):
        def generator():
          for i in range(n):
              yield element
        return Stream(generator())

    def buffered_mix(self, elements_in_buffer_count):
        def generator():
            buffer = ListRandom()
            it = iter(self)
            while True:
                while len(buffer) < elements_in_buffer_count:
                    try:
                        buffer.add_item(next(it))
                    except StopIteration:
                        while len(buffer) > 0:
                            yield buffer.pop_random()
                        return
                yield buffer.pop_random()
        return Stream(generator())


    @staticmethod
    def mix_streams(streams, weights):
        def generator():
            iters = [iter(i) for i in streams]
            choices = list(range(len(streams)))
            i = 0
            while True:
                try:
                    i = random.choices(choices, weights)[0]
                    yield next(iters[i])
                except StopIteration:
                    weights[i] = 0
                    if sum(weights) == 0:
                        return
        return Stream(generator())


    def chain(self, another_stream):
        def generator():
            for i in self:
                yield i
            for i in another_stream:
                yield i
        return Stream(generator())

    def slide_window(self, window_size):
        res = deque()
        for i in self:
          res.append(i)
          if len(res) == window_size:
            yield Stream(res)
            res.popleft()

    def skip(self, count):
        def generator():
            n = count
            for i in self.generator:
                n -= 1
                if n == 0: break
            for i in self.generator:
                yield i
        return Stream(generator())

    def get(self, count):
        res = []
        for i in self:
            res.append(i)
            if len(res) == count:
                return res
        return res

    def limit(self, count):
        def generator():
            n = count
            for i in self.generator:
                yield i
                n -= 1
                if n == 0: break
        return Stream(generator())

    def map(self, func):
        def generator():
            for i in self.generator:
                yield func(i)
        return Stream(generator())

    def starmap(self, func):
        def generator():
            for i in self.generator:
                for j in func(i):
                    yield j
        return Stream(generator())

    def group(self, n):
        def generator():
            grouped = []
            for i in self.generator:
                grouped.append(i)
                if len(grouped) >= n:
                    yield grouped
                    grouped = []
            if len(grouped) != 0:
                yield grouped

        return Stream(generator())



import functools
from collections import deque
import random
random.seed(42)

@functools.lru_cache(maxsize=128)
def get_word_features_cached(word):
    return get_word_features(word, params)#.numpy()

class Substr:
    def __init__(self, text):
        self.text = text
    def __repr__(self) -> str:
        return f"Substring(-1, -1, {self.text})"

def d_as_str(d):
  return "<" + " ".join(map(lambda text: text.text, d))+ ">"


async def infer_optimal(params, text):
  # print("INFERCENC IS WIERD\n" * 10)
  res = []
  last_inserted_pos = 0
  def sink(token, log=False):
    nonlocal last_inserted_pos
    if token.text == "PAD": return
    if log: print('sink', token)
    if isinstance(token, Substr):
      res.append(token.text)
      if log: print("added1 ", f"`{token.text}`", token)
    else:
      if last_inserted_pos != token.start:
        res.append(text[last_inserted_pos: token.start])
        if log: print("added2 ", f"`{text[last_inserted_pos: token.start]}`", last_inserted_pos, token.start)
      last_inserted_pos = token.stop
      res.append(token.text)
      if log: print("added3 ", f"`{token.text}`", token)

  def skip(token, log=False):
    nonlocal last_inserted_pos
    last_inserted_pos = token.stop
    if log: print('skip', token)

  def sink_remaining():
     res.append(text[last_inserted_pos:])


  async def predict_on_tokens(window_left, window_right, return_probas):
    features = [get_word_features_cached(i.text) for i in Stream(window_left).chain(window_right)]
    features_for_batch = np.stack((features, ))
    arr = np.ascontiguousarray(features_for_batch, dtype=np.float32)
    output_probas = np.array((await jsinfer.infer(arr)).to_py())
    # output_probas[0][0] += 2.
    if return_probas:
      return params["ID_TO_PUNCTUATION"], output_probas
    punct_idx = np.argmax(output_probas).item()
    punct = params["ID_TO_PUNCTUATION"][punct_idx]
    return punct


  window_left = deque()
  window_right = deque()
  log = False
  skip_next = False
  for i in Stream.repeat(Substr(PAD_TOKEN), params['INPUT_WORDS_CNT_LEFT']) \
      .chain(Stream(tokenize(text))) \
      .chain(Stream.repeat(Substr(PAD_TOKEN), params["INPUT_WORDS_CNT_RIGHT"])):
    window_right.append(i)
    if len(window_right) <= params["INPUT_WORDS_CNT_RIGHT"]:
        continue
    assert len(window_right) == params["INPUT_WORDS_CNT_RIGHT"] + 1

    next_ = window_right.popleft()
    sink(next_)
    window_left.append(next_)
    if len(window_left) < params['INPUT_WORDS_CNT_LEFT']:
      continue

    assert len(window_left) == params["INPUT_WORDS_CNT_LEFT"]
    assert len(window_right) == params["INPUT_WORDS_CNT_RIGHT"]

    if skip_next or window_right[0].text in '?!':
      prediction = "$skip"
    else:
      # params["ID_TO_PUNCTUATION"], output_probas
      prediction = await predict_on_tokens(window_left, window_right, return_probas=False)


    #random.choice([" ", "."])
    if log: print(d_as_str(window_left).rjust(100), prediction.center(6), d_as_str(window_right))

    def is_replaceable_punct(punct):
      return punct in ',.'

    if prediction == "$skip":
      pass
    elif prediction != "$empty":
      if is_replaceable_punct(window_right[0].text):
        if window_right[0].text != prediction:
          window_right[0].text = prediction
      else:
        window_left.append(Substr(prediction))
        sink(window_left[-1])
    else:
      if is_replaceable_punct(window_right[0].text):
          skip(window_right.popleft())

    skip_next = is_replaceable_punct(window_right[0].text) or window_right[0].text in '?!'

    while len(window_left) != params['INPUT_WORDS_CNT_LEFT'] - 1:
      token = window_left.popleft()

    if log: print(d_as_str(window_left).rjust(100), "      ", d_as_str(window_right))

  for i in window_right:
    sink(i)
  sink_remaining()
  ress = "".join(res)
  return ress