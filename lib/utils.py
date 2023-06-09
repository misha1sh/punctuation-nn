
from joblib import Parallel, delayed
import time
from tqdm.notebook import tqdm
import numpy as np
import torch
# import matplotlib.pyplot as plt
# import librosa
import os
import urllib
import zipfile

root = "./cache"
if not os.path.exists(root):
    os.makedirs(root)

class ProgressParallel(Parallel):
    def __init__(self, use_tqdm=True, total=None, *args, **kwargs):
        self._use_tqdm = use_tqdm
        self._total = total
        super().__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        with tqdm(disable=not self._use_tqdm, total=self._total) as self._pbar:
            return Parallel.__call__(self, *args, **kwargs)

    def print_progress(self):
        if self._total is None:
            self._pbar.total = self.n_dispatched_tasks
        self._pbar.n = self.n_completed_tasks
        self._pbar.refresh()


def download_file(file, url):
    file_path = os.path.join(root, file)
    if os.path.isfile(file_path):
        return file_path
    print("donwloading", file, "to", file_path)
    # url = BASE_URL + file
    urllib.request.urlretrieve(url, file_path)
    return file_path

def unzip_file(file_path):
    unzipped_dir = os.path.splitext(file_path)[0]
    if os.path.isdir(unzipped_dir):
        return unzipped_dir

    print("extracting", file_path)
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(root)
    return unzipped_dir


def plot_waveform(waveform, sr, title="Waveform"):
    try:
        waveform = waveform.numpy()
    except:
        waveform = np.array(waveform)

    if len(waveform.shape) == 1:
        waveform = np.array([waveform])

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    figure, axes = plt.subplots(num_channels, 1)
    axes.plot(time_axis, waveform[0], linewidth=1)
    axes.grid(True)
    figure.suptitle(title)
    plt.show(block=False)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin"):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto")
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)

def plot_mel_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")
    plt.show(block=False)



def reverse_dict(d, priority_for_duplicates=[]):
    res = {val: key for key, val in d.items()}
    for key in priority_for_duplicates:
        if key not in d: continue
        res[d[key]] = key
    return res

def run_proc(task):
    import multiprocessing
    print("starting proc")
    p = multiprocessing.Process(target=task)
    try:
        p.start()
        p.join()
    except KeyboardInterrupt:
        print("terminating proccess")
        p.terminate()
        from time import sleep
        sleep(1)
        print("killing")
        p.kill()


def chunks(multiple_lists, chunk_size):
    for i in range(0, len(multiple_lists[0]), chunk_size):
        yield tuple(l[i:i + chunk_size] for l in multiple_lists)
def size_of_tensor(tensor):
    return round(tensor.nelement() * tensor.element_size())
def count_parameters(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / 1024 ** 2
# SPEECH_WAVEFORM = wavs[0]
# plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform")
# melspec = spectrogrammer.wave2mel(Tensor(SPEECH_WAVEFORM))
# plot_spectrogram(melspec, title="MelSpectrogram - torchaudio", ylabel="mel freq")
# Audio([SPEECH_WAVEFORM], rate=SAMPLE_RATE)