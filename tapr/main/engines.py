from abc import ABC, abstractmethod
from concurrent import futures as ft
from dask.distributed import Client

class Engine(ABC):
    @abstractmethod
    def __tapr_engine_map__(self, func, *args):
        pass

class ProcessEngine(Engine):
    def __init__(self, processes):
        self._processes = processes

    @property
    def processes(self):
        return self._processes

    def __tapr_engine_map__(self, func, *args):
        with ft.ProcessPoolExecutor(self._processes) as ex:
            return list(ex.map(func, *args))

    def __str__(self):
        return f"Process Engine\nProcesses: {self._processes}"

    def __repr__(self):
        return str(self)


class ThreadEngine(Engine):
    def __init__(self, threads):
        self._threads = threads

    @property
    def threads(self):
        return self._threads

    def __tapr_engine_map__(self, func, *args):
        with ft.ThreadPoolExecutor(self._threads) as ex:
            return list(ex.map(func, *args))

    def __str__(self):
        return f"Thread Engine\nThreads: {self._threads}"

    def __repr__(self):
        return str(self)


class StandardEngine(Engine):
    def __init__(self):
        pass

    def __tapr_engine_map__(self, func, *args):
        return list(map(func, *args))

    def __str__(self):
        return "Standard (serial) Engine"

    def __repr__(self):
        return str(self)


class DaskClientEngine(Engine):
    def __init__(self, *client_args, **client_kwargs):
        self._client = Client(*client_args, **client_kwargs)

    @property
    def client(self):
        return self._client

    def __tapr_engine_map__(self, func, *args):
        args = [list(arg) for arg in args]
        futures = self._client.map(func, *args)
        return self._client.gather(futures)

    def __str__(self):
        return f"Dask Client Engine"

    def __repr__(self):
        return str(self)