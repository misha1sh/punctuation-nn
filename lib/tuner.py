import optuna

class ForcedTrial:
    def __init__(self, trial: optuna.Trial):
        self.trial = trial
        self.overrides = {}
        self.saved_values = {}

    def report(self, value, step):
        return self.trial.report(value, step)

    def set_system_attr(self, key, value):
        return self.trial.set_system_attr(key, value)

    def set_user_attr(self, key, value):
        return self.trial.set_user_attr(key, value)

    def should_prune(self):
        return self.trial.should_prune()

    def suggest_categorical(self, name, choices):
        if name in self.overrides:
            value = self.overrides[name]
        else:
            value = self.trial.suggest_categorical(name, choices)
        self.saved_values[name] = value
        return value

    # def suggest_discrete_uniform(self, name, low, high, q):
    #     if name in self.overrides:
    #         value = self.overrides[name]
    #     else:
    #         value = self.trial.suggest_discrete_uniform(name, low, high, q)
    #     self.saved_values[name] = value
    #     return value

    def suggest_float(self, name, default, low, high, *, step=None, log=False):
        if name in self.overrides:
            value = self.overrides[name]
        else:
            value = self.trial.suggest_float(name, default, low, high, step=step, log=log)
        self.saved_values[name] = value
        return value

    def suggest_int(self, name, default, low, high, step=1, log=False):
        if name in self.overrides:
            value = self.overrides[name]
        else:
            value = self.trial.suggest_int(name, default, low, high, step=step, log=log)
        self.saved_values[name] = value
        return value

    # def suggest_loguniform(self, name, low, high):
    #     if name in self.overrides:
    #         value = self.overrides[name]
    #     else:
    #         value = self.trial.suggest_loguniform(name, low, high)
    #     self.saved_values[name] = value
    #     return value

    # def suggest_uniform(self, name, low, high):
    #     if name in self.overrides:
    #         value = self.overrides[name]
    #     else:
    #         value = self.trial.suggest_uniform(name, low, high)
    #     self.saved_values[name] = value
    #     return value

    def override(self, name, value):
        self.overrides[name] = value

    def get_saved_values(self):
        return self.saved_values

    @property
    def datetime_start(self):
        return self.trial.datetime_start

    @property
    def distributions(self):
        return self.trial.distributions

    @property
    def number(self):
        return self.trial.number

    @property
    def params(self):
        return self.trial.params

    @property
    def system_attrs(self):
        return self.trial.system_attrs

    @property
    def user_attrs(self):
        return self.trial.user_attrs


class SavedTrial:
    def __init__(self, saved_values):
        self.saved_values = saved_values

    def suggest_categorical(self, name, choices):
        if name in self.saved_values:
            return self.saved_values[name]
        else:
            raise ValueError(f"Value for parameter '{name}' not found in saved values.")

    def suggest_float(self, name, low, high, *, step=None, log=False):
        if name in self.saved_values:
            return self.saved_values[name]
        else:
            raise ValueError(f"Value for parameter '{name}' not found in saved values.")

    def suggest_int(self, name, low, high, step=1, log=False):
        if name in self.saved_values:
            return self.saved_values[name]
        else:
            raise ValueError(f"Value for parameter '{name}' not found in saved values.")


class TrialWrapper():
    def __init__(self, trial: optuna.Trial):
        self.trial = trial

    def suggest_categorical(self, *args, **kwargs):
        return self.trial.suggest_categorical(*args, **kwargs)

    def suggest_int(self, name, default, *args, **kwargs):
        return self.trial.suggest_int(name, *args, **kwargs)

    def suggest_float(self, name, default, *args, **kwargs):
        return self.trial.suggest_float(name, *args, **kwargs)

    def report(self, *args, **kwargs):
        return self.trial.report(*args, **kwargs)

    def should_prune(self):
        return self.trial.should_prune()

    @property
    def params(self):
        return self.trial.params

class TunedParams:
    def __init__(self, params):
        self.params = params

    def suggest_categorical(self, name, choices):
        if name in self.params:
            return self.params[name]

        return choices[0]

    def suggest_int(self, name, default, *args, **kwargs):
        if name in self.params:
            return self.params[name]
        return default

    def suggest_float(self, name, default, *args, **kwargs):
        if name in self.params:
            return self.params[name]
        return default

    # unused
    def report(self, *args, **kwargs): pass
    def set_system_attr(self, *args, **kwargs): pass
    def set_user_attr(self, *args, **kwargs): pass
    def should_prune(self): return False

storage_path = "sqlite:///tune/opt.db"

def load_best(name):
    study = optuna.load_study(study_name=name, storage=storage_path)
    trial = study.best_trial
    return TunedParams(trial.params)

def tune(name, objective, n_trials, timeout, try_continue=False):
    study_loaded = False
    if try_continue:
        try:
            study = optuna.load_study(study_name=name, storage=storage_path) #, pruner=pruner)
            study_loaded = True
        except Exception as ex:
            print("Failed to load study", name, ex)

    if not study_loaded:
        sampler = optuna.samplers.TPESampler()
        study = optuna.create_study(study_name=name, sampler=sampler, direction="minimize", storage=storage_path) #, pruner=pruner)

    def wrapped_objective(trial):
        return objective(TrialWrapper(trial))

    try:
        study.optimize(wrapped_objective, n_trials=n_trials, timeout=timeout)
    except KeyboardInterrupt:
        print("Tuning interrupted")

    print("Number of finished trials: {}".format(len(study.trials)))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    return TunedParams(study.best_trial.params)


