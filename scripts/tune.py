import tuner
import multiprocessing
from storage import Storage

from repr import train_model, Model, get_train_params_from_trial

def model_runner(model, train_params, path, result):
  # result.value = 10
  train_model(model, train_params, path, result2=result)
  print("finsihed!")

def objective(trial):
  trial = tuner.ForcedTrial(trial)

  trial.overrides = {'INTERNAL_EMBEDDING_SIZE': 256, 'INTERNAL_EMBEDDING_SIZE2': 16, 'encoder_count': 4, 'encoder_num_heads': 8, 'encoder_residual_norm_style': 'pre', 'lstm_layers': 0, 'optimizer': 'Yogi', 'pre_linear_count': 0, 'optimizer_beta1': 0.5, 'optimizer_eps': 1e-8, 'optmizer_beta2': 0.999,
                     'optimizer_weight_decay': 0.,
                     'optimizer_initial_accumulator': 1e-6}
  result = manager.Value('result', 1.)
  model = Model(params, trial)

  train_params = get_train_params_from_trial(trial, 2000)
  print(trial.params)
  process = ctx.Process(target=model_runner, args=(model, train_params, path, result,))
  process.start()
  process.join()

  return result.value

# change patinece back???
if __name__ == '__main__':
    ctx = multiprocessing.get_context('spawn')
    manager = ctx.Manager()
    path = "cache2/storage2"
    storage = Storage(path)
    params = storage.wait_meta_change("params", None)
    # tuner.tune("writers_tune_24", objective, 100000, 100000)

    tuner.tune("scheduler_tune_5", objective, 100000, 100000)

    # for opt in  ['RAdam', "PID", "Yogi", 'Adam', 'Adam.amsgrad', 'Lamb']:
    #    tuner.tune("tune_8_" + opt, objective, 20, 100000, try_continue=True)