import re







def isin(collection):
    def function(value):
        return value in collection

    return function


def contains(string):
    def function(value):
        return string in value

    return function


def matches(pattern):
    def function(value):
        return bool(re.match(pattern, value))

    return function


class NTableFilter:
    def __init__(self, ntbl):
        self._ntbl = ntbl

    def __getitem__(self, index):
        ntbl = self._ntbl
        for k, v in index.items():
            # v should be a function that returns True or False based on some
            # condition on the items of the iterable (in this case the keys of
            # the dimension mapping)
            findex = list(filter(v, ntbl.ntable_map(k)))
            ntbl = ntbl.ntable_map(k)[findex]

        return ntbl

    def __call__(self, **kwargs):
        return self[kwargs]
