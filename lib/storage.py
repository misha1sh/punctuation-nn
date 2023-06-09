from pathlib import Path
from collections import defaultdict
import dill
import torch
import os
import asyncio
import shutil
import fcntl
import pyinotify
import threading

class Locked:
    def __init__(self, file_path, mode):
        self.file_path = file_path
        self.mode = mode
    
    def __enter__(self):
        self.file_lock = str(self.file_path) + ".lock"
        self.lockfile = open(self.file_lock, "w")
        fcntl.flock(self.lockfile, fcntl.LOCK_EX)
        self.file = open(self.file_path, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()
        fcntl.flock(self.lockfile, fcntl.LOCK_UN)
        self.lockfile.close()



class FileEventHandler(pyinotify.ProcessEvent):
    def __init__(self, filepath):
        super().__init__()
        self.file_events = {}
        self.wm = pyinotify.WatchManager()
        self.mask = pyinotify.IN_MODIFY | pyinotify.IN_CREATE
        self.notifier = pyinotify.Notifier(self.wm, self)
        self.wdds = self.wm.add_watch(filepath, self.mask, rec=False)
        self.t1 = threading.Thread(target=self.notifier.loop, daemon=True)
        self.t1.start()

    def process_IN_MODIFY(self, event):
        for filepath, event_obj in self.file_events.items():
            if str(event.pathname).endswith(str(filepath)):
                event_obj.set()

    def process_IN_CREATE(self, event):
        for filepath, event_obj in self.file_events.items():
            if str(event.pathname).endswith(str(filepath)):
                event_obj.set()

    def wait_for_file_event(self, filepath):
        event_obj = self.file_events.setdefault(filepath, threading.Event())
        event_obj.clear()
        while not event_obj.is_set():
            event_obj.wait()
        event_obj.clear()



class Storage:
    def __init__(self, path, enable_watcher=True):
        self.path = Path(path)
        if enable_watcher:
            self.handler = FileEventHandler([path])
        os.makedirs(self.path, exist_ok=True)

    def clear(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        self.chunks_count = defaultdict(int)
        self.future = None
        os.makedirs(self.path, exist_ok=True)

    def store(self, name, chunk_id, data): 
        chunk_id = str(chunk_id)
        os.makedirs(self.path / chunk_id, exist_ok=True)
        
        if isinstance(data, torch.Tensor):
            file_ext = '.pt'
        else:
            file_ext = '.pickle'

        with open(self.path / chunk_id / f'{name}_{chunk_id}{file_ext}', 'wb') as file:
            if isinstance(data, torch.Tensor):
                torch.save(data, file)
            else:
                dill.dump(data, file)


    def _find_file(self, name, chunk_id):
        filepath = str(self.path / str(chunk_id) / f'{name}_{chunk_id}')
        if os.path.isfile(filepath + '.pickle'):
            return filepath + '.pickle', dill.load
        elif os.path.isfile(filepath + '.pt'):
            return filepath + '.pt', torch.load
        else:
            raise ValueError(f"Unsupported file format: {filepath}")

    def _read_file(self, name, chunk_id, *args, **kwargs):
        filepath, method = self._find_file(name, chunk_id)
        with open(filepath, "rb") as file:
            return method(file, *args, **kwargs)

    def get(self, name, chunk_id, *args, **kwargs):
        return self._read_file(name, chunk_id, *args, *kwargs)


    def write_meta(self, name, meta):
        with Locked(self.path / str(name), "wb") as file:
            dill.dump(meta, file)
    
    def wait_meta_change(self, name, reference_meta):
        while True:
            try:
                new_meta = self.get_meta(name) 
                if dill.dumps(new_meta) != dill.dumps(reference_meta):
                    return new_meta   
            except FileNotFoundError:
                pass
   
            self.handler.wait_for_file_event(self.path / str(name))


    def get_meta(self, name):
        with Locked(self.path / str(name), "rb") as file:
            return dill.load(file)
