import dataset_builder
import zipfile
import importlib
import gc
from utils import download_file
from corus import load_lenta2
import torch
from utils import ProgressParallel, chunks, size_of_tensor, count_parameters
from stream import Stream
# from disklist import DiskList

importlib.reload(dataset_builder)
# input, output = create_dataset([
#     'Однако самые ранние свидетельства приручения кошек древними египтянами относились к 2000—1900 годам до нашей эры. А недавно было установлено, что в Древний Египет домашняя кошка попала, как и на Кипр, с территории Анатолии. В настоящее время кошка является одним из самых популярных домашних животных. ',
#     'В лесу родилась елочка, в лесу она росла.'])
# input.shape, output.shape

def lenta_path():
    return download_file("lenta-ru-news.csv.gz",
    "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2")
def nerus_file():
    return download_file("nerus_lenta.conllu.gz", "https://storage.yandexcloud.net/natasha-nerus/data/nerus_lenta.conllu.gz")

def get_lenta_records():
    return load_lenta2(lenta_path())

def read_lenta_records(records, cnt):
    res = []
    for record in records:
        res.append(record.text)
        if len(res) >= cnt: break
    return res

# https://www.kaggle.com/datasets/d0rj3228/russian-literature
writers_zip_path = "./cache2/writers.zip"

def read_writers():
    res = []
    with zipfile.ZipFile(writers_zip_path, 'r') as writers_dir:
        for file in writers_dir.filelist:
            f = file.filename
            good_file = ('prose' in f or 'publicism' in f) and 'txt' in f and 'Blok' not in f # and 'Tolstoy' in f
            if not good_file: continue
            with writers_dir.open(f) as file_open:
                res.append(file_open.read().decode())
    return res

# def get_writers_file_streams():
#     streams = []
#     with zipfile.ZipFile(writers_zip_path, 'r') as writers_dir:
#         for file in writers_dir.filelist:
#             f = file.filename
#             good_file = ('prose' in f or 'publicism' in f) and 'txt' in f and 'Blok' not in f # and 'Tolstoy' in f
#             if not good_file: continue
#             streams.append(writers_dir.open(f))

#             # TODO: support file closing
#             # with writers_dir.open(f) as file_open:
#                 # streams
#                 # yield file_open.read().decode()

def split_into_parts(text):
    part = ""
    for line in text.split("\n"):
        if len(line.split()) > 6 and \
            len(set(line) & set("ёйцукенгшщзхъфывапролджэячсмитьбю")) > 0:
            part += line
        elif len(part) > 0:
            yield part
            part = ""

    if part != "":
        yield part

def get_writers_multi_stream():
    data = read_writers()
    sizes = [len(i) for i in data]
    streams = [Stream(split_into_parts(i)).buffered_mix(1024) for i in data]
    return Stream.mix_streams(streams, sizes)


# def get_writers_records_stream():
#     stream = Stream(get_writers_records())
#     return stream.starmap(split_into_parts)

def get_nerus_records():
    return load_nerus(nerus_file)

def read_nerus_records(records, cnt):
    res = []
    for record in records:
        for sent in record.sents:
            res.append(sent)
            if len(res) >= cnt: return res
    return res
# res = read_writers()
# sum(map(len, res)),sum(map(len, read_lenta_records(10000)))


def concat_lists(lists):
    return [item for sublist in lists for item in sublist]

# @mem_cached("create_lenta_dataset")
# @file_cached("create_dataset")
def create_dataset(cnt, parts, params):
    if params['type'] == "lenta":
        records = get_lenta_records()
        method = read_lenta_records
    # elif params['type'] == "writers":
    #     texts = read_writers()
    elif params['type'] == "nerus":
        records = get_nerus_records()
        method = read_nerus_records

    dataset_res = []
    for i in range(parts):
        gc.collect()
        print("part", i, "/", parts)
        texts = method(records, cnt // parts)
        rows = dataset_builder.create_dataset(texts, params)
        if len(dataset_res) == 0:
            dataset_res = list(rows)
        else:
            for i in range(len(dataset_res)):
                if isinstance(rows[i], list):
                    dataset_res[i] = concat_lists((rows[i], dataset_res[i], ))
                else:
                    dataset_res[i] = torch.cat((rows[i], dataset_res[i]))

    return {"input": dataset_res[0],
            "output": dataset_res[1],
            "texts_res": dataset_res[2],
            "is_infected": dataset_res[3]}

# params['type'] = 'lenta' #'nerus'
# dataset = create_dataset(15000, 2, params) # 150_000 # 600_000
# gc.collect()
# input, output = dataset['input'], dataset['output']
# input.shape, output.shape, len(dataset['texts_res']), dataset['is_infected'].shape, dataset['is_infected'].sum()


class Dataset:
    def __init__(self, params, train_test_split, chunk_size, batch_size):
        self.params = params
        self.train_test_split = 0.9
        self.chunk_size = chunk_size
        self.batch_size = batch_size

        self.x_train_chunks = []
        self.y_train_chunks = []

        self.x_test = torch.FloatTensor()
        self.y_test = torch.FloatTensor()

        # self.texts_res = DiskList(cache_size=1, tmp_dir="./cache/")
        self.is_infected = list()

        self.train_indices = torch.LongTensor()
        self.test_indices = torch.LongTensor()

    def iter_train_batches(self):
        for x_train_chunk, y_train_chunk in zip(self.x_train_chunks, self.y_train_chunks):
            x_train_chunk_gpu = x_train_chunk.to(self.device)
            for x, y in zip(torch.split(x_train_chunk_gpu, self.batch_size),
                            torch.split(y_train_chunk, self.batch_size)):
                yield (x, y)
            del x_train_chunk_gpu

    def split_into_train_test(self, dataset):
        for x, y, text_res, is_infected in chunks(dataset, self.chunk_size // 3):
            total_len = x.shape[0]
            train_len = int(self.train_test_split * total_len)
            test_len = int(total_len - train_len)
            x_train, x_test = torch.utils.data.random_split(x, [train_len, test_len])

            only_train_indices = set(torch.arange(total_len)[is_infected].numpy())
            train_indices = torch.LongTensor(list(set(x_train.indices) | only_train_indices))
            test_indices = torch.LongTensor(list(set(x_test.indices) - only_train_indices))

            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            current_length = len(self.train_indices) + len(self.test_indices)
            # self.texts_res.extend(text_res)
            self.is_infected.extend(is_infected)
            self.train_indices = torch.cat((self.train_indices, train_indices + current_length))
            self.test_indices = torch.cat((self.test_indices, test_indices + current_length))

            if self.x_train_chunks and \
                    len(self.x_train_chunks[-1]) + len(x_train) < self.chunk_size:
                self.x_train_chunks[-1] = torch.cat((self.x_train_chunks[-1], x_train))
                self.y_train_chunks[-1] = torch.cat((self.y_train_chunks[-1], y_train))
            else:
                self.x_train_chunks.append(x_train)
                self.y_train_chunks.append(y_train)

            self.x_test = torch.cat((self.x_test, x_test))
            self.y_test = torch.cat((self.y_test, y_test))


    def load(self, cnt, cnt_per_part):
        raw_data_iter = get_lenta_records()
        load_raw_data = read_lenta_records

        cur_cnt = 0
        for i in range(cnt // cnt_per_part + 1):
            print(f"part [{cur_cnt}/{cnt}]\r")
            cnt_to_load = min(cnt - cur_cnt, cnt_per_part)
            dataset_raw_data = load_raw_data(raw_data_iter, cnt_to_load)
            cur_cnt += len(dataset_raw_data)

            dataset_part = dataset_builder.create_dataset(dataset_raw_data, self.params)
            self.split_into_train_test(dataset_part)


        stats = {'x_size_mb': 0., 'train_cnt': 0,
                 'y_size_mb': 0, 'test_cnt': 0}
        for x_train_chunk, y_train_chunk in zip(self.x_train_chunks, self.y_train_chunks):
            stats['x_size_mb'] += size_of_tensor(x_train_chunk) / 1024 / 1024
            stats['y_size_mb'] += size_of_tensor(y_train_chunk) / 1024 / 1024
            stats['train_cnt'] += x_train_chunk.shape[0]
        stats['test_cnt'] = self.x_test.shape[0]
        stats['chunks'] = len(self.x_train_chunks)
        stats['mb_per_chunk'] = stats['x_size_mb'] / stats['chunks']

        print("Train: {train_cnt} samples (x={x_size_mb:.2f} Mb, y={y_size_mb:.2f} Mb, {chunks} chunks, {mb_per_chunk:.2f} Mb per chunk)".format(**stats))
        print("Test: {test_cnt} samples".format(**stats))

    def to_gpu(self):
        self.device = torch.device('cuda:0')
        self.y_train_chunks = [y_train_chunk.to(self.device) for y_train_chunk in self.y_train_chunks]
        self.x_test = self.x_test.pin_memory()
        self.y_test = self.y_test.to(self.device)
        import gc
        for i in range(len(self.x_train_chunks)):
            gc.collect()
            self.x_train_chunks[i] = self.x_train_chunks[i].pin_memory()
