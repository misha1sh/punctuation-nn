import torch
import torch.nn as nn


import optuna
import math

import time
import psutil
import gc

def sample(sample_count, x):
    perm = torch.randperm(x.size(0))
    idx = perm[:sample_count]
    return x[idx]

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
# class PreheatSampler:
#     def __init__(self, initial_count, grow_amount, patience):
#         self.samples_count = initial_count
#         self.grow_amount = grow_amount
#         self.overfit_detector = EarlyStopping(patience=patience)


#     def update_loss(self, test_loss, time_passed):
#         if self.overfit_detector.need_stop(test_loss):
#             self.overfit_detector.reset()
#             self.samples_count += self.grow_amount

#     def sample(self, x):
#         return x[:self.samples_count]

# class PreheatSamplerPerTime:
#     def __init__(self, initial_count, grow_amount):
#         self.initial_count  = initial_count
#         self.samples_count = initial_count
#         self.grow_amount = grow_amount

#     def update_loss(self, test_loss, time_passed):
#         self.samples_count = self.initial_count + math.ceil(self.grow_amount * time_passed)

#     def sample(self, x):
#         # return x[:self.samples_count]
#         return sample(self.samples_count, x)



# class PercentSampler:
#     pass

class NoSampler:
    def update_loss(self, test_loss, time_passed): pass
    def sample(self, x):
        return x


class EarlyStopping:
    def __init__(self, patience):
        self.patience = patience
        self.ticks_since_best_loss = 0
        self.best_loss = None


    def need_stop(self, loss):
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
            self.ticks_since_best_loss = 0
            return False

        self.ticks_since_best_loss += 1
        if self.ticks_since_best_loss > self.patience:
            return True

    def reset(self):
        self.best_loss = None
        self.ticks_since_best_loss = 0


class EarlyStoppingPerTime:
    def __init__(self, patience):
        self.patience = patience
        self.ticks_since_best_loss = 0
        self.best_loss = None


    def need_stop(self, loss, time_passed):
        if self.best_loss is None or self.best_loss > loss:
            self.best_loss = loss
            self.ticks_since_best_loss = time_passed
            return False

        self.ticks_since_best_loss += 1
        if time_passed > self.ticks_since_best_loss + self.patience:
            return True

    def reset(self):
        self.best_loss = None

class Trainer:
    def __init__(self, model, loss, optimizer, scheduler, additional_losses,
            warmup_scheduler=None, sampler=NoSampler(), enable_chunking=False,
            schedule_every_minibatch=False): # calc_loss_on_data
        self.device = torch.device('cuda:0')
        self.model = model.to(self.device)
        self.loss = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.schedule_every_minibatch = schedule_every_minibatch
        self.warmup_scheduler = warmup_scheduler
        self.sampler = sampler
        # self.calc_loss_on_data = calc_loss_on_data
        self.early_stop_lambda = lambda trainer, test_loss, time_passed, log: False
        self.additional_losses = additional_losses
        self.history = {
            "train_loss": [],
            "test_loss": [],
        }
        self.additional_test_loss = lambda y_real, y_pred, is_infected, epoch: None
        self.enable_chunking = enable_chunking
        # self.history.update({i: [] for i in additional_losses})

    def set_data(self, dataset):
        self.dataset = dataset

    def calc_loss_on_data_internal(self, x_real, y_real, return_pred = False):
        y_pred = self.model(x_real)
        err = self.loss(y_pred, y_real)
        if return_pred:
            return err, y_pred
        return err

    def train_batch(self, x_real, y_real):
        self.model.zero_grad()
        err = self.calc_loss_on_data_internal(self.sampler.sample(x_real), y_real)
        err.backward()
        self.optimizer.step()
        # if self.scheduler:
        #     self.scheduler.step(err)
        return err.item()

    def predict(self, x):
        return self.model(x.to(self.device))

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def train(self, epochs, batch=None, chunk_size=None, result2=None, trial=None, log=True, writer = None):
        assert (self.enable_chunking and chunk_size) or (not self.enable_chunking and not chunk_size)

        report_period = (epochs // 4)
        start_time = time.time()

        miniepoch = 0

        train_batch_len = None

        for epoch in range(epochs):
            self.model.train()

            cnt_trained = 0
            train_losses = []
            while cnt_trained < 100_000:
                for x, y in self.dataset.iter_train_batches(miniepoch):
                    if train_batch_len is None: train_batch_len = x.shape[0]
                    train_losses.append(self.train_batch(x, y))
                    cnt_trained += x.shape[0]
                    miniepoch += 1
                    if self.schedule_every_minibatch and self.scheduler:
                        self.scheduler.step()
                        if writer:
                            writer.add_scalar('1 Misc/Learning rate (minibatch)', get_lr(self.optimizer), miniepoch)
                            writer.add_scalar('! Loss/train (minibatch)', train_losses[-1], miniepoch)
                del x
                del y

            train_loss = sum(train_losses) / len(train_losses)

            self.model.eval()
            with torch.no_grad():
                test_loss = []
                y_test_pred = []
                print(self.dataset.x_test.shape[0])
                for x, y in zip(torch.split(self.dataset.x_test, train_batch_len),
                                torch.split(self.dataset.y_test, train_batch_len)):
                    test_loss_batch, y_test_pred_batch = \
                        self.calc_loss_on_data_internal(x, y, return_pred=True)
                    test_loss.append(test_loss_batch.item())
                    y_test_pred.append(y_test_pred_batch)
                del x
                del y
                del y_test_pred_batch

                test_loss = sum(test_loss) / len(test_loss)
                y_test_pred = torch.cat(y_test_pred)
                self.additional_test_loss(self.dataset.y_test, y_test_pred,
                                          self.dataset.is_infected_test, epoch)
                del y_test_pred

            # TRASH
            if self.scheduler and not self.schedule_every_minibatch and miniepoch < epochs:
                if self.warmup_scheduler:
                    with self.warmup_scheduler.dampening():
                        self.scheduler.step(test_loss)
                else:
                    self.scheduler.step(test_loss)


            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)

            time_passed = time.time() - start_time
            self.sampler.update_loss(test_loss, time_passed)

            if writer is not None:
                writer.add_scalar('! Loss/train', train_loss, epoch)
                writer.add_scalar('! Loss/test', test_loss, epoch)
                writer.add_scalar('1 Misc/Learning rate', get_lr(self.optimizer), epoch)

            if log:
                additional_losses = ""
                for loss_name in self.additional_losses:
                    loss_dict = self.additional_losses[loss_name](self)
                    for loss_name2 in loss_dict:
                        name = loss_name + "." + loss_name2
                        loss = loss_dict[loss_name2]
                        if name not in self.history:
                            self.history[name] = []
                        self.history[name].append(loss)
                        additional_losses += loss_name2 + ": %.4f " % loss

                print('[%d] [%d/%d] [%.1f s]\t loss: %.4f loss_test: %.4f  lr: %.4f  %s\r'
                    % (epoch, miniepoch, epochs, time_passed,
                        train_loss, test_loss, self.get_lr(), additional_losses))

            print("EARLY STOP DISABLED!!. TRASH INSTEAD OF EPOCHS!!")
            # if get_lr(self.optimizer) < 3e-5:
            #     print("stopped because of low learning rate")
            #     return

            if self.early_stop_lambda(self, test_loss, time_passed,log):
                if log: print("Early stop at ", epoch, " epoch")
                break

            # if trial2 and (epoch % report_period == report_period - 1):
            #     trial2.put((test_loss, epoch))

            if trial and (epoch % report_period == report_period - 1):
                trial.report(test_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            # TRASH
            if miniepoch > epochs:
                if log: print("miniepoch STOP")
                break

        if result2:
          result2.value = test_loss

    def plot_history(self, cutoff=100):
        import matplotlib.pyplot as plt
        plt.style.use('fivethirtyeight')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')

        plt.plot(self.history['train_loss'][cutoff:], label='train_loss')
        plt.plot(self.history['test_loss'][cutoff:], label='test_loss')
        plt.legend()