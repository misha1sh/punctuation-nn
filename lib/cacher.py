import os
import numpy as np
import hashlib
MEM_CACHE = {}
root = os.fspath("/home/misha-sh/audio-ml/cache/")

def get_args_hash(args, kwargs):
    key = str(args + tuple(sorted(kwargs.items())))
    res = hashlib.sha256(key.encode('utf-8')).hexdigest()
    # print(args, kwargs, " as str: `" + key +  "`", "res: ", res)
    return res

def file_cached(name):
    file_path = os.path.join(root, "cache", name)
    def decorator(func):
        def wrapper(*args, **kwargs):
            file_path_arg = file_path + "$$$" + get_args_hash(args, kwargs)

            if os.path.isfile(file_path_arg):
                with open(file_path_arg, "rb") as file:
                    return np.load(file, allow_pickle=True)[0]

            res = func(*args, **kwargs)

            with open(file_path_arg, "wb") as file:
                np.save(file, [res], allow_pickle=True)

            return res

        return wrapper
    return decorator

def mem_cached(name):
    if name not in MEM_CACHE:
        MEM_CACHE[name] = {}
    def decorator(func):
        def wrapper(*args, **kwargs):
            name_arg = get_args_hash(args, kwargs)
            if name not in MEM_CACHE:
                MEM_CACHE[name] = {}
            if name_arg in MEM_CACHE[name]:
                return MEM_CACHE[name][name_arg]

            res = func(*args, **kwargs)
            MEM_CACHE[name][name_arg] = res

            return res
        return wrapper
    return decorator

def clear_cache(name):
    for file_path in glob.glob(os.path.join(root, "cache", name) + "$$$*"):
        if os.path.isfile(file_path):
            os.remove(file_path)

    if name in MEM_CACHE:
        MEM_CACHE[name] = {}
