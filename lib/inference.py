from dataset_builder import calculate_word_features_for_tokens, get_word_features, PAD_TOKEN
import onnxruntime as ort
import numpy as np
import torch
import joblib

def torch_model_runner(model):
    model.eval()
    def func(input):
        with torch.no_grad():
            return model(input)
    return func

def onnx_model_runner(path):
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = joblib.cpu_count()
    ort_sess = ort.InferenceSession(path, sess_opt)
    def func(input):
        return ort_sess.run(None, {'input': np.array(input) })[0]
    return func

def infer(params, model, text):
    assert params["RETAIN_LEFT_PUNCT"] #

    unpadded_tokens = text.split(' ')
    unpadded_tokens = list(filter(lambda x: len(x) > 0, unpadded_tokens))
    tokens = [PAD_TOKEN] * params['INPUT_WORDS_CNT_LEFT'] + unpadded_tokens + [PAD_TOKEN] * (params["INPUT_WORDS_CNT_RIGHT"] + 1)
    features = calculate_word_features_for_tokens(tokens, params).float()

    res = ""

    i = params['INPUT_WORDS_CNT_LEFT']
    while i < len(tokens) - params['INPUT_WORDS_CNT_RIGHT']:
        tokens_for_batch = tokens[i - params['INPUT_WORDS_CNT_LEFT']: i + params['INPUT_WORDS_CNT_RIGHT']]

        tokens_for_batch_copy = tokens_for_batch.copy()
        tokens_for_batch_copy.insert(params['INPUT_WORDS_CNT_LEFT'], '?')
        # print(" ".join(tokens_for_batch_copy))


        features_for_batch = features[i - params['INPUT_WORDS_CNT_LEFT']: i + params['INPUT_WORDS_CNT_RIGHT']]
        features_for_batch = torch.stack((features_for_batch, ))
        output_probs = model(features_for_batch)
        punct_idx = np.argmax(output_probs).item()
        punct = params["ID_TO_PUNCTUATION"][punct_idx]

        # print(punct)

        # punct = '.'

        if punct != '$empty':
            res += punct
            if tokens[i] != 'PAD':
                res += " " + tokens[i]
            tokens.insert(i, punct)
            features = torch.cat((features[:i],
                    torch.stack((get_word_features(punct, params), )),
                    features[i:]), 0)
            i += 2
        else:
            if tokens[i] != 'PAD':
                res += " " + tokens[i]
            i += 1

    return res.strip()
