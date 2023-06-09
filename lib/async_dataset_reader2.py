from dataclasses import dataclass
import torch
from storage import Storage
import threading
import time

LOG = False

@dataclass
class DatasetChunk:
    i: int
    x: torch.Tensor
    y: torch.Tensor
    in_memory: bool
    last_read: int
    used: int

    def __repr__(self):
        return f"DatasetChunk {self.i} in_memory={self.in_memory} last_read={self.last_read} used={self.used}"

class AsyncDatasetLoaderToGPU:
    def __init__(self, async_dataset_reader, max_kept_in_memory, test_samples_count, writer):
        global async_dataset_to_gpu_loader_thread

        assert max_kept_in_memory > 1

        self.async_dataset_reader = async_dataset_reader
        self.time_counter = 0

        self.keep_running = True

        self.max_kept_in_memory = max_kept_in_memory
        self.lock = threading.Lock()
        self.first_loaded_event = threading.Event()
        self.loaded_event = threading.Event()
        self.chunks = [] # List of DatasetChunk

        self.last_read = -1

        self.test_samples_count = test_samples_count

        self.x_test, self.y_test = None, None

        self.writer = writer

        self.batch_size = self.async_dataset_reader.storage.get_meta("params")["batch_size"]

        if LOG: print("[gpu] loading test")
        self.x_test = async_dataset_reader.storage.get_meta("x_test").float().cuda()
        self.y_test = async_dataset_reader.storage.get_meta("y_test").float().cuda()
        self.text_res_test = async_dataset_reader.storage.get_meta("text_res_test")
        self.is_infected_test = async_dataset_reader.storage.get_meta("is_infected_test")
        if LOG: print("[gpu] test loaded", self.x_test.shape)

        async_dataset_to_gpu_loader_thread = threading.Thread(target=self.read_infinitely)
        async_dataset_to_gpu_loader_thread.start()

    def stop(self):
        self.keep_running = False

    def wait_for_first_chunk(self):
        self.async_dataset_reader.first_loaded_event.wait()

    def find_candidates(self, in_memory):
        stats = list(filter(lambda chunk: chunk.in_memory == in_memory, self.chunks))
        stats.sort(key=lambda chunk: chunk.used * 1000 + chunk.last_read)
        # print(f"priorites({in_memory}) ", stats)
        return stats

    def count_in_memory(self):
        return sum(map(lambda chunk: chunk.in_memory, self.chunks))

    def has_used(self):
        return any(map(lambda chunk: chunk.used > 0 and chunk.in_memory, self.chunks))

    def read_next(self):
        self.loaded_event.clear()

        self.chunks_count = self.async_dataset_reader.chunks_count
        if self.count_in_memory() == self.chunks_count:
            if LOG: print("[gpu] loaded all into memory")
            # wait till next chunk is written
            self.storage.wait_meta_change("chunks_count", self.chunks_count)
            return

        if self.count_in_memory() >= self.max_kept_in_memory:
            with self.lock:
                if self.has_used():
                    chunk_to_remove = self.find_candidates(in_memory=True)[-1]
                    if chunk_to_remove.used == 0:
                        print(self.find_candidates(in_memory=True))
                    assert chunk_to_remove.used > 0
                    x, y = chunk_to_remove.x, chunk_to_remove.y
                    chunk_to_remove.x, chunk_to_remove.y = None, None
                    del x
                    del y
                    chunk_to_remove.in_memory = False
                    chunk_to_remove.used = 0
                    if LOG: print("[gpu] removed ", chunk_to_remove.i)
                else:
                    time.sleep(0.2)
                    return
        res = self.async_dataset_reader.get_chunk_on_gpu(
            set(map(lambda chunk: chunk.i,
                filter(lambda chunk: chunk.in_memory, self.chunks))))
        if res is None:
            if LOG: print("[gpu] loaded all into memory(2)")
            time.sleep(0.2)
            return

        chunk_i, x, y = res
        if LOG: print(f"[gpu] reading {chunk_i}")

        for i in range(len(self.chunks), chunk_i + 1):
            self.chunks.append(DatasetChunk(i=i, x=None, y=None,
                                            in_memory=False,
                                            last_read=0, used=0))
        candidate = self.chunks[chunk_i]

        candidate.x = x
        candidate.y = y

        # if candidate.i == 0:
        #     split_left_x, split_right_x = candidate.x[:self.test_samples_count], candidate.x[self.test_samples_count:]
        #     split_left_y, split_right_y = candidate.y[:self.test_samples_count], candidate.y[self.test_samples_count:]
        #     if self.x_test == None:
        #         self.x_test = split_left_x
        #         self.y_test = split_left_y

        #         is_infected = self.async_dataset_reader.storage.get("is_infected", i)
        #         # infected_indices = set(torch.arange(self.x_test.shape[0])[is_infected].numpy())
        #         # test_indices = torch.LongTensor(list(set(torch.arange(self.x_test.shape[0])) -
        #         #                                         infected_indices))

        #         # self.test_indices = ~(is_infected[:self.test_samples_count])
        #         self.is_infected_test = is_infected[:self.test_samples_count].clone()
        #         self.x_test = self.x_test[:self.test_samples_count].clone()
        #         self.y_test = self.y_test[:self.test_samples_count].clone()

        #     candidate.x = split_right_x
        #     candidate.y = split_right_y

        #.to(self.device) #, map_location=device)
        with self.lock:
            candidate.in_memory = True
            candidate.used = 0
        if LOG: print(f"[gpu] read {chunk_i}")
        self.first_loaded_event.set()
        self.loaded_event.set()

    def read_infinitely(self):
        self.wait_for_first_chunk()
        while self.keep_running:
            self.read_next()

    def iter_train_batches(self, epoch):
        self.first_loaded_event.wait()

        waited = 0
        while True:
            with self.lock:
                candidate = self.find_candidates(in_memory=True)[0]
                if candidate.used <= 2: break
            self.loaded_event.wait()
            if LOG: print("[gpu] WAITED BECAUSE OF OVERUSE")
            waited += 1

        self.writer.add_scalar('GPU/Wait because of overuse', waited, epoch)


        with self.lock:
            candidate = self.find_candidates(in_memory=True)[0]
            if LOG: print("[gpu] iter", candidate.i)
            x_train_chunk = candidate.x#.to(self.device, dtype=torch.float32, non_blocking=True)
            y_train_chunk = candidate.y#.to(self.device, dtype=torch.float32, non_blocking=True)

            if self.writer is not None:
                self.writer.add_scalar('GPU/Chunk id', candidate.i, epoch)
                self.writer.add_scalar('GPU/Reuse', candidate.used, epoch)

            for x, y in zip(torch.split(x_train_chunk, self.batch_size),
                            torch.split(y_train_chunk, self.batch_size)):
                yield (x, y)

            candidate.used += 1
            self.time_counter += 1
            candidate.last_read = self.time_counter












class AsyncDatasetReader:
    def __init__(self, path, max_kept_in_memory, writer=None):
        global async_dataset_reader_thread

        assert max_kept_in_memory > 1

        self.keep_running = True
        self.storage = Storage(path)
        self.time_counter = 0

        self.max_kept_in_memory = max_kept_in_memory
        self.lock = threading.Lock()
        self.first_loaded_event = threading.Event()
        self.chunks = [] # List of DatasetChunk

        self.last_read = -1
        self.device = torch.device("cuda:0")

        self.writer = writer

        async_dataset_reader_thread = threading.Thread(target=self.read_infinitely)
        async_dataset_reader_thread.start()

    def stop(self):
        self.keep_running = False

    def wait_for_first_chunk(self):
        self.params = self.storage.wait_meta_change("params", None)
        self.chunks_count = self.storage.wait_meta_change("chunks_count", 0)

    def find_candidates(self, in_memory, sort_by_used=False):
        stats = list(filter(lambda chunk: chunk.in_memory == in_memory, self.chunks))
        if sort_by_used:
            sort_by = lambda chunk: chunk.used
        else:
            sort_by = lambda chunk: chunk.last_read
        stats.sort(key=sort_by)
        # if in_memory:
        #     print(f"[cpu] priorites({in_memory}) ", stats)
        return stats

    def count_in_memory(self):
        return sum(map(lambda chunk: chunk.in_memory, self.chunks))

    def has_used(self):
        return any(map(lambda chunk: chunk.used and chunk.in_memory, self.chunks))

    def read_next(self):
        self.chunks_count = self.storage.get_meta("chunks_count")
        for i in range(len(self.chunks), self.chunks_count):
            self.chunks.append(DatasetChunk(i=i, x=None, y=None,
                                            in_memory=False,
                                            last_read=0, used=0))


        if self.count_in_memory() == self.chunks_count:
            if LOG: print("loaded all into memory")
            # wait till next chunk is written
            self.storage.wait_meta_change("chunks_count", self.chunks_count)
            return

        if self.count_in_memory() == self.max_kept_in_memory:
            if self.has_used():
                with self.lock:
                    chunk_to_remove = self.find_candidates(in_memory=True, sort_by_used=True)[-1]
                    assert chunk_to_remove.used > 0
                    x, y = chunk_to_remove.x, chunk_to_remove.y
                    chunk_to_remove.x, chunk_to_remove.y = None, None
                    del x
                    del y
                    chunk_to_remove.in_memory = False
                    chunk_to_remove.used += 1
                    if LOG: print("removed ", chunk_to_remove.i)
            else:
                time.sleep(0.2)
                return

        candidate = self.find_candidates(in_memory=False)[0]
        i = candidate.i
        if LOG: print(f"reading {i} (last_read={candidate.last_read})")
        candidate.x = self.storage.get("x", i).to(dtype=torch.float32).pin_memory()
        #.to(self.device) #, map_location=device)
        candidate.y = self.storage.get("y", i).to(dtype=torch.float32).pin_memory()
        #.to(self.device) #, map_location=device)
        with self.lock:
            candidate.in_memory = True
            candidate.used = 0
        if LOG: print(f"read {i}")
        self.first_loaded_event.set()

    def read_infinitely(self):
        self.wait_for_first_chunk()
        while self.keep_running:
            self.read_next()

    def get_chunk_on_gpu(self, chunks_to_ignore):
        self.first_loaded_event.wait()

        with self.lock:
            candidates = self.find_candidates(in_memory=True)
            for candidate in candidates:
                if candidate.i in chunks_to_ignore:
                    continue
                break
            else:
                return None

            i = candidate.i
            if LOG: print("iter", candidate.i)
            try:
                x_train_chunk = candidate.x.to(self.device, non_blocking=True) #, dtype=torch.float32, non_blocking=True)
                y_train_chunk = candidate.y.to(self.device, non_blocking=True) #, dtype=torch.float32, non_blocking=True)
            except Exception as e:
                print("some exeception during loading into memory", e)
                return None

            candidate.used += 1
            candidate.last_read = self.time_counter
            if self.writer:
                self.writer.add_scalar('CPU/Chunk id', candidate.i, self.time_counter)
                self.writer.add_scalar('CPU/Reuse', candidate.used, self.time_counter)
            self.time_counter += 1

        return i, x_train_chunk, y_train_chunk
