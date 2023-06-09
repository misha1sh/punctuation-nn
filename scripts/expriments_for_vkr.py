import tuner
import multiprocessing
from storage import Storage
from torch import nn

from repr import train_model, Model, get_train_params_from_trial

def model_runner(model, train_params, path, result):
  # result.value = 10
  train_model(model, train_params, path, result2=result)
  print("finsihed!")


class LinearModel(nn.Module):
    def __init__(self, params, trial, **kwargs):
        super().__init__()
        N_words = params['INPUT_WORDS_CNT']
        N_features = params['TOTAL_WORD_FEATURES_CNT']
        # input is (N, N_words, N_features)
        # output is (N, N_words, )
        self.N_features = N_features

        self.model = nn.Sequential(*[
            # (N, N_words, N_features + ...)
            nn.Linear(N_features, 128),
            nn.ReLU(),
            nn.Flatten(1),
            nn.Linear(N_words* 128, N_words* 64),
            nn.Tanh(),
            nn.Linear(N_words* 64, N_words* 64),
            nn.ReLU(),
            nn.Linear(N_words* 64, N_words* 64),
            nn.ReLU(),
            nn.Linear(N_words* 64, N_words* 64),
            nn.ReLU(),
            nn.Linear(N_words* 64, N_words* 64),
            nn.ReLU(),
            nn.Linear(N_words* 64, N_words* 64),
            nn.ReLU(),
            nn.Linear(N_words* 64, N_words* 32),
            nn.ReLU(),
            nn.Linear(N_words* 32, params['TARGET_CLASSES_COUNT']),
        ])


    def forward(self, x):
        # print(x.shape)
        # print(nn.Flatten(1)(nn.Linear(self.N_features, 128).cuda()(x)).shape)
        return self.model(x)

class LSTMModel(nn.Module):
    class LSTM(nn.Module):
        def __init__(self, cnt, num_layers, **kwargs):
            super().__init__()
            self.lstm = nn.LSTM(cnt,
                                cnt // 2,
                                num_layers=num_layers,
                                batch_first=True, bidirectional=True)
        def forward(self, x):
            return self.lstm(x)[0]

    def __init__(self, params, trial, **kwargs):
        super().__init__()
        N_words = params['INPUT_WORDS_CNT']
        N_features = params['TOTAL_WORD_FEATURES_CNT']
        # input is (N, N_words, N_features)
        # output is (N, N_words, )
        self.N_features = N_features

        self.model = nn.Sequential(*[
            # (N, N_words, N_features + ...)
            nn.Linear(N_features, 256),
            nn.ReLU(),

            self.LSTM(256, 4),
            nn.Linear(256, 256),
            nn.ReLU(),
            self.LSTM(256, 4),

            nn.Flatten(1),
            nn.Linear(N_words* 256, N_words* 32),
            nn.ReLU(),
            nn.Linear(N_words* 32, params['TARGET_CLASSES_COUNT']),
        ])


    def forward(self, x):
        return self.model(x)



def train_separate_process(model, trial, count):
#   trial = tuner.ForcedTrial(trial)
#   model = LinearModel(params, trial)
  train_params = get_train_params_from_trial(trial, count)
  result = manager.Value('result', 1.)
  print(trial.params)
  process = ctx.Process(target=model_runner, args=(model, train_params, path, result,))
  process.start()
  process.join()
  return result.value

# change patinece back???
if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()
    path = "cache2/storage_no_morph"
    # path = "cache2/storage2"
    print("NO PUNCT")
    storage = Storage(path)
    params = storage.wait_meta_change("params", None)

    mult = 1
    lr = 0.001
    if "no_morph" in path:
        lr = 0.0005
        mult = 5
    trial = tuner.TunedParams( {'INTERNAL_EMBEDDING_SIZE': 512,
                                'INTERNAL_EMBEDDING_SIZE2': 256,
                               'encoder_count': 6, 'encoder_num_heads': 16, 'encoder_residual_norm_style':
                               'pre', 'lr': lr, 'lstm_layers': 2,
                               'lstm_macro_layers': 2, 'optimizer': 'RAdam', 'pre_linear_count':
                               0, 'opt': 'Yogi', 'optimizer_beta1': 0.9,
                            'cycle': 998 * mult, 'lr': lr, 'min_lr': 5e-05,
                            'scheduler_gamma': 0.7019020081089732,
                            'warmup_steps': 400 * mult})
                            #'warmup_steps': 257})
    model = LSTMModel(params, trial)
    # model = Model(params, trial)
    # model = LinearModel(params, trial)
    # from utils import count_parameters
    # print(round(count_parameters(model), 3), "Mb of parameters")
    # path = "cache2/storage2"
    # storage = Storage(path)
    # params = storage.wait_meta_change("params", None)
    # print(params)

    # asdfasdf
    train_separate_process(model, trial, 50000)
    # tuner.tune("writers_tune_24", objective, 100000, 100000)

    # tuner.tune("scheduler_tune_5", objective, 100000, 100000)

    # for opt in  ['RAdam', "PID", "Yogi", 'Adam', 'Adam.amsgrad', 'Lamb']:
    #    tuner.tune("tune_8_" + opt, objective, 20, 100000, try_continue=True)