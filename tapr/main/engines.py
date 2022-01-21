from concurrent import futures as ft


class ProcessEngine:
    def __init__(self, processes):
        self._processes = processes

    @property
    def processes(self):
        return self._processes

    def __call__(self, func, *args):
        with ft.ProcessPoolExecutor(self._processes) as ex:
            return list(ex.map(func, *args))

    def __str__(self):
        return f"Process Engine\nProcesses: {self._processes}"

    def __repr__(self):
        return str(self)


class ThreadEngine:
    def __init__(self, threads):
        self._threads = threads

    @property
    def threads(self):
        return self._threads

    def __call__(self, func, *args):
        with ft.ThreadPoolExecutor(self._threads) as ex:
            return list(ex.map(func, *args))

    def __str__(self):
        return f"Thread Engine\nThreads: {self._threads}"

    def __repr__(self):
        return str(self)


class StandardEngine:
    def __init__(self):
        pass

    def __call__(self, func, *args):
        return list(map(func, *args))

    def __str__(self):
        return "Standard (serial) Engine"

    def __repr__(self):
        return str(self)
