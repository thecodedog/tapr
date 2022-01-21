import warnings as wn


class _CallAndHandle:
    def __init__(self, handler, func):
        self._handler = handler
        self._func = func

    def __call__(self, *args, **kwargs):
        try:
            return self._func(*args, **kwargs)
        except Exception as e:
            return self._handler(e, self._func, *args, **kwargs)


def handled_by(handler):
    def decorator(func):
        return _CallAndHandle(handler, func)

    return decorator


class FunctionError:
    def __init__(self, e, func, *args, **kwargs):
        self._e = e
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        return self._func(*self._args, **self._kwargs)

    def __str__(self):
        string = ""
        string += f"Exception:\n{self._e}\n"
        string += f"Function:\n{self._func}\n"
        string += f"Arguments:\n{self._args}\n"
        string += f"Key-Word Arguments:\n{self._kwargs}\n"
        return string

    def __repr__(self):
        return str(self)

    def __bool__(self):
        return False

    def __eq__(self, other):
        result = type(self._e) == type(other._e)
        result &= self._e.args == other._e.args
        result &= self._func == other._func
        result &= self._args == other._args
        result &= self._kwargs == other._kwargs
        return result


def print_warning_return_function_error(e, func, *args, **kwargs):
    wn.warn(
        f"Warning: the function {func.__name__} raised a {type(e)} exception when given arguments {args} and kwargs {kwargs}"
    )
    return FunctionError(e, func, *args, **kwargs)
