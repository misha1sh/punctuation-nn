import xformers
import torch
# print(torch.ops.xformers.matmul_with_mask(torch.Tensor([[1, 2], [3, 4]]),
#                                         torch.Tensor([[1, 2], [3,4]]),
#                                         torch.Tensor([[True, False], [False, False]])))

# res = torch.matmul(torch.Tensor([[1, 2], [3, 4]]), torch.Tensor([[1, 2], [3,4]]))
# res[torch.Tensor([[True, False], [False, False]]).bool().logical_not()] =-float("Inf")
# print(res)
# exit(0)

import dill
import torch
from storage import Storage
prefix = "results_big_model_GOOD_2022-05-15"
print("ADD RESULTS ")
with open(f"{prefix}/some_model_CLASS.dill", "rb") as file:
    Model = dill.load(file)
model = torch.load(f"{prefix}/some_model.pt", map_location=torch.device('cpu')) #).v1", map_location=torch.device('cuda:0'))

with open(f"{prefix}/storage_path.dill", "rb") as file:
    storage_path = dill.load(file)


storage = Storage(str(storage_path))
params = storage.get_meta("params")
x_test = storage.get("x", 0).float()
print(x_test.shape)

dummy_input = x_test[0:1]
print(dummy_input.shape)



from xformers.components.attention.attention_patterns import (
    local_1d_pattern)
for i in [3, 4, 5, 6, 7]:
    att = list(model.children())[0][i].wrap_att.layer.sublayer.children().__next__()
    # att._get_local_mask = lambda shape: local_1d_pattern(shape[1], att.window_size * 3).to_sparse()
    att.attention_mask = None

# torch.jit.script(model, example_inputs=[(dummy_input, )])

with torch.no_grad():
    print(model(dummy_input))
    torch.onnx.export(model, (dummy_input, ), f"{prefix}/model.onnx",
                    input_names = ['input'],
                    output_names = ['output'],
                      #verbose=True
                      )




import numpy as np
import random
import pymorphy3
import numpy as np
import math
import pickle
from razdel import tokenize

from dataset_builder import calculate_word_features_for_tokens, PAD_TOKEN,get_word_features
from inference import torch_model_runner, onnx_model_runner, infer

# print("PREFIX\n" * 100)

# onnx_model = onnx_model_runner("results writers big/model.onnx")
onnx_model = onnx_model_runner(f"{prefix}/model.onnx")
with open("params.pickle", "rb") as f:
    params = pickle.load(f)

print(onnx_model(dummy_input.numpy()))

class jsinfer:
    async def infer(arr):
        class wrapper:
            def to_py():
                return onnx_model(arr)
        return wrapper

from stream import Stream
import functools
from collections import deque
import random
random.seed(42)

@functools.lru_cache(maxsize=128)
def get_word_features_cached(word):
    return get_word_features(word, params).numpy()

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

    if skip_next:
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

    skip_next = is_replaceable_punct(window_right[0].text)

    while len(window_left) != params['INPUT_WORDS_CNT_LEFT'] - 1:
      token = window_left.popleft()

    if log: print(d_as_str(window_left).rjust(100), "      ", d_as_str(window_right))

  for i in window_right:
    sink(i)
  sink_remaining()
  ress = "".join(res)
  return ress




from collections import defaultdict
text = "Тест. тест, тест. Тест"
text_res = "Тест. тест, тест. Тест."
# text_res = await infer_optimal(params, text)

def calculate_diff2(text, text_res):
  res = defaultdict(lambda: 0)

  def is_punctuation(c):
      return c in ".,"

  def sink_add(c):
    nonlocal res
    res['added ' + c] += 1

  def sink_remove(c):
    nonlocal res
    res['removed ' + c] += 1

  def sink_change(c1, c2):
    nonlocal res
    res['changed ' + c1 + " with " + c2] += 1

  i = 0
  j = 0
  while True:
      if i >= len(text): break
      if j >= len(text_res): break
      # print(text[i], text_res[j])
      if text[i] == text_res[j]:
          if is_punctuation(text[i]):
             res['not changed ' + text[i]] += 1
          i += 1
          j += 1
          continue

      if is_punctuation(text[i]) and is_punctuation(text_res[j]):
        sink_change(text[i], text_res[j])
        i += 1
        j += 1
        continue

      if is_punctuation(text[i]):
        sink_remove(text[i])
        i += 1
        continue

      if is_punctuation(text_res[j]):
        sink_add(text_res[j])
        j += 1
        continue

      raise Exception("Change not in punctuation", text[i], text_res[j], "at ", i, j)

  while i < len(text):
    # print("remaining: ", text[i])
    assert is_punctuation(text[i])
    sink_remove(text[i])
    i += 1

  while j < len(text_res):
    # print("remaining(2): ",text_res[j])
    assert is_punctuation(text_res[j])
    sink_add(text_res[j])
    j += 1

  res['possible punctuation places'] = len(list(tokenize(text)))

  return res






print(calculate_diff2(text, text_res))
# print(infer_optimal(params, "кек\n"))


import glob
from striprtf.striprtf import rtf_to_text
from tqdm.notebook import tqdm



def dicts_sum(dict1, dict2):
  for key in dict2:
    dict1[key] += dict2[key]
  return dict1

async def task(clear_punctuation):
    res = defaultdict(lambda: 0)
    i = 0
    for rtf_path in tqdm(glob.glob("../validation/Mark Tven/Mark Tven rtf/*.rtf")):
        with open(rtf_path, "rb") as rtf_file:
            encoded = rtf_file.read()
            try:
                rtf = encoded.decode('cp1251')
                txt = rtf_to_text(rtf)
                if clear_punctuation:
                   text_to_infer = txt.replace(". ", " ").replace(", ", " ")
                else:
                   text_to_infer = txt

                diff = calculate_diff2(txt, await infer_optimal(params, text_to_infer))
                res = dicts_sum(res, diff)
            except Exception as ex:
                print("skipped ", rtf_path, len(encoded), ex)
            # raise
            i += 1
            # if i> 2: break

    print(res)


import asyncio
loop = asyncio.get_event_loop()
for clear_punctuation in [False, True]:
    print("clear_punctuation", clear_punctuation)
    task_to_wait = loop.create_task(task(clear_punctuation)),
    loop.run_until_complete(asyncio.wait(task_to_wait))
loop.close()







