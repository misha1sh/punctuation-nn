from dataclasses import dataclass
import torch
from storage import Storage
import threading
import time

@dataclass
class DatasetChunk:
    i: int
    x: torch.Tensor
    y: torch.Tensor
    in_memory: bool
    last_read: int
    used: bool

    def __repr__(self):
        return f"DatasetChunk {self.i} in_memory={self.in_memory} last_read={self.last_read} used={self.used}"


class AsyncDatasetReader:
    def __init__(self, path, max_kept_in_memory, test_samples_count):
        global async_dataset_reader_thread

        assert max_kept_in_memory > 1

        self.storage = Storage(path)
        self.time_counter = 0
                
        self.max_kept_in_memory = max_kept_in_memory
        self.lock = threading.Lock()
        self.first_loaded_event = threading.Event()
        self.chunks = [] # List of DatasetChunk

        self.last_read = -1

        self.device = torch.device("cuda:0")

        self.test_samples_count = test_samples_count

        self.x_test, y_test = None, None

        async_dataset_reader_thread = threading.Thread(target=self.read_infinitely)
        async_dataset_reader_thread.start()

    def wait_for_first_chunk(self):
        self.params = self.storage.wait_meta_change("params", None)
        self.chunks_count = self.storage.wait_meta_change("chunks_count", 0)

    def find_candidates(self, in_memory):
        stats = list(filter(lambda chunk: chunk.in_memory == in_memory, self.chunks))
        stats.sort(key=lambda chunk: chunk.last_read)
        # print(f"priorites({in_memory}) ", stats)
        return stats
    
    def count_in_memory(self):
        return sum(map(lambda chunk: chunk.in_memory, self.chunks))

    def has_used(self):
        return any(map(lambda chunk: chunk.used and chunk.in_memory, self.chunks))

    def read_next(self):
        self.chunks_count = self.storage.get_meta("chunks_count")
        for i in range(len(self.chunks), self.chunks_count):
            self.chunks.append(DatasetChunk(i, None, None, False, 0, False))
        
      
        if self.count_in_memory() == self.chunks_count:
            print("loaded all into memory")
            # wait till next chunk is written
            self.storage.wait_meta_change("chunks_count", self.chunks_count)
            return
        
        if self.count_in_memory() == self.max_kept_in_memory:
            if self.has_used():
                with self.lock:
                    chunk_to_remove = self.find_candidates(in_memory=True)[-1]
                    assert chunk_to_remove.used
                    x, y = chunk_to_remove.x, chunk_to_remove.y
                    chunk_to_remove.x, chunk_to_remove.y = None, None
                    del x
                    del y
                    chunk_to_remove.in_memory = False
                    chunk_to_remove.used = False
                    print("removed ", chunk_to_remove.i)
            else:
                time.sleep(0.2)
                return

        candidate = self.find_candidates(in_memory=False)[0]
        i = candidate.i
        print(f"reading {i} (last_read={candidate.last_read})")
        candidate.x = self.storage.get("x", i).to(self.device, dtype=torch.float32, non_blocking=True)
        #.to(self.device) #, map_location=device)
        candidate.y = self.storage.get("y", i).to(self.device, dtype=torch.float32, non_blocking=True)

        if candidate.i == 0:
            split_left_x, split_right_x = candidate.x[:self.test_samples_count], candidate.x[self.test_samples_count:]
            split_left_y, split_right_y = candidate.y[:self.test_samples_count], candidate.y[self.test_samples_count:]
            if self.x_test == None:
               self.x_test = split_left_x.clone()
               self.y_test = split_left_y.clone()
            candidate.x = split_right_x
            candidate.y = split_right_y

        #.to(self.device) #, map_location=device)
        with self.lock:
            candidate.in_memory = True
            candidate.used = False
        print(f"read {i}")
        self.first_loaded_event.set()
        
    def read_infinitely(self):
        self.wait_for_first_chunk()
        while True:
            self.read_next()

    def iter_train_batches(self):
        self.first_loaded_event.wait()

        with self.lock:
            candidate = self.find_candidates(in_memory=True)[0]
            print("iter", candidate.i)
            x_train_chunk = candidate.x#.to(self.device, dtype=torch.float32, non_blocking=True)
            y_train_chunk = candidate.y#.to(self.device, dtype=torch.float32, non_blocking=True)

            candidate.used = True    
            self.time_counter += 1
            candidate.last_read = self.time_counter

        for x, y in zip(torch.split(x_train_chunk, self.params["batch_size"]),
                        torch.split(y_train_chunk, self.params["batch_size"])):
            yield (x, y)

        del x_train_chunk
        del y_train_chunk
