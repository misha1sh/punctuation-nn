import socket
import dill
import concurrent
import io
import queue
import threading
import select

# class FileAsyncResult:
#     def __init__(self, filepath):
#         self.event = threading.Event()
#         self.filepath = filepath
#         self.exception = None
        
#     def set_exception(self, exception):
#         self.exception = exception
#         self.event.set()

#     def set_result(self, sockfile):
#         with open(self.filepath) as file:
#             size = dill.load(sockfile)
#             while size > 0:
#                 cnt_read = min(size, 10 * 1024 * 1024)
#                 file.write(sockfile.read(cnt_read))
#                 size -= cnt_read
#         self.event.set()

class BasicAsyncResult:
    def __init__(self):
        self.event = threading.Event()
        self.result = None
        self.exception = None
        self.callback = None
        self.completed = False

    def _complete(self):
        self.event.set()
        if self.callback is not None:
            self.callback(self)
        self.completed = True

    def set_exception(self, exception):
        assert not self.completed
        self.exception = exception
        self._complete()
        
    def set_result(self, sockfile):
        assert not self.completed
        self.result = dill.load(sockfile)
        self._complete()
        
    def get_result(self):
        if self.exception is not None:
            raise self.exception
        return self.result

    def set_callback(self, callback):
        assert callable(callback)
        self.callback = callback

    def __repr__(self) -> str:
        if not self.completed:
            return f"BasicAsyncResult not completed"
        if self.exception is not None:
            return f"BasicAsyncResult completed with exception `{self.exception}`"
        return f"BasicAsyncResult completed with result `{self.result.__repr__()[:20]}...`"
        
class RemoteRunnerServer:
    def __init__(self):
        self.requests_queue = queue.Queue()

    def connection_handler(self, sock):
        with sock, sock.makefile('rwb') as sockfile:
            while True:
                try:
                    result, typ, rpc_request = self.requests_queue.get(timeout=1)
                except queue.Empty:
                    # Heartbeat
                    sockfile.write(b'h')
                    sockfile.flush()

                    # Wait 3 seconds
                    ready_to_read, _, _ = select.select([sock], [], [], 5)
                    if sockfile in ready_to_read:
                        print("heartbeat timeout. Closing connection")
                        return

                    res = sockfile.read(1)
                    if res != b'h':
                        print("Heartbeat failed. Closing connection")
                        return
                    continue
                # GLOBALS
                # sockfile.write(b'0')
                # dill.dump({
                #     "func4": func4
                # }, sockfile)
            
                # RPC COMMAND
                sockfile.write(typ) # b'1'
                dill.dump(rpc_request, sockfile)                          
                sockfile.flush()
                
                typ = sockfile.read(1)
                if typ == b'0':
                    result.set_result(sockfile)
                elif typ == b'1':
                    result.set_exception(dill.load(sockfile))
                elif typ == b'':
                    print("disconnected")
                    return
                else:
                    raise Exception("unknown type " + str(typ))
    
    def send_globals(self, globls):
        assert isinstance(globls, dict)
        
        result = BasicAsyncResult()
        self.requests_queue.put((result, b'0', globls))
        result.event.wait()
        return result.result     
    
    def rpc_async(self, func, *args, **kwargs):
        assert callable(func)
                                
        result = BasicAsyncResult()
        self.requests_queue.put((result, b'1', ((func, args, kwargs, 1))))
        return result
    
    def rpc_simple(self, func, *args, **kwargs):
        assert callable(func)
                                
        result = BasicAsyncResult()
        self.requests_queue.put((result, b'1', ((func, args, kwargs, 1))))
        result.event.wait()
        if result.exception:
            raise result.exception
        return result.result
        
    # def rpc_file(self, file, func, *args, **kwargs):
    #     assert callable(func)
    #     assert isinstance(file, str)
    #     result = FileAsyncResult(file)
    #     self.requests_queue.put((result, b'1', ((func, args, kwargs, 1))))
    #     result.event.wait()
    #     if result.exception:
    #         raise result.exception
    #     return True
    
    def host_server(self):
        HOST = '0.0.0.0' 
        PORT = 65231

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            # listen for incoming connections
            s.listen()
            print(f"Server is listening on {HOST}:{PORT}")
            # wait for a client to connect
            
            with concurrent.futures.ThreadPoolExecutor(4, "ServerConnectionHandler") as executor:
                while True:
                    conn, addr = s.accept()
                    print(f"Connected by {addr}")
                    self.connection_handler(conn)
                    # executor.submit(self.handle_connection, conn)
                        # receive data from the client
    
    def run(self):
        self.thread = threading.Thread(target=self.host_server)
        self.thread.start()

def is_sendable(x):
    if callable(x):
        return True
    if str(type(x)) != "<class 'module'>":
        return False
    return 'built-in' not in str(x) 

def get_sendable_globals(globs):
    res = {}
    for k,v in globs.items():
        if k in ['exit', 'open', 'quit', 'get_ipython']: continue
        if not is_sendable(v): continue
        res[k] = v
    return res

server = None
def run_server_if_not_running():
    global server
    try:
        def func(): return 1
        assert server.rpc_simple(func) == 1
        print("server already running")
        return server
    except:
        server = RemoteRunnerServer()
        server.run()
        return server


import os
import glob
def server_install_packages(server):
    os.system('python -W ignore setup.py -q bdist_egg >/dev/null 2>&1')
    with open(glob.glob('dist/*.egg')[0], "rb") as f:
        package_data = f.read()

    #def package_install(package_data):
    package_install = f"""
global egg_file
import tempfile
import pkgutil
import importlib
import sys
egg_file = tempfile.NamedTemporaryFile(suffix=".egg")
egg_file.write(package_data)
egg_file.flush()
if egg_file.name not in sys.path:
    sys.path.insert(0, egg_file.name)
for i in list(pkgutil.iter_modules([egg_file.name])):
    importlib.reload(importlib.import_module(i.name))
    print("loaded ", i.name)
    """

    server.rpc_simple(exec, package_install, {"package_data": package_data})
    server.rpc_simple(exec, "from imports import *")
    # server.rpc_simple(exec, "import importlib\nimport dataset_builder\nimportlib.reload(dataset_builder)")

