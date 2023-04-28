import pickle as pk
import json
import sys
import io

import numpy as np
import plotly as pl

from ..main.utils import NULL
from ..main.engines import StandardEngine, ProcessEngine, ThreadEngine

DEFAULT_TYPE_ID = "__TAPR_DEFAULT_TYPE__"

serializers = {}
deserializers = {DEFAULT_TYPE_ID:pk.loads}

def serializer(type_, id_):
    def decorator(func):
        serializers[type_] = (func, id_)

    return decorator

def deserializer(id_):
    def decorator(func):
        deserializers[id_] = func

    return decorator

def serialize(obj, allow_pickle=False):
    try:
        srlzr, type_id = serializers[type(obj)]
    except KeyError:
        srlzr, type_id = pk.dumps, DEFAULT_TYPE_ID

    if srlzr == pk.dumps:
        if not allow_pickle:
            raise ValueError(f"Non-pickle serializer for type {type(obj)} not found and allow_pickle flag was False. Either define a non-pickle serializer or set the flag to True.")
    return srlzr(obj), type_id

def deserialize(bytes_, id_, allow_pickle=False):
    try:
        dsrlzr = deserializers[id_]
    except KeyError:
        dsrlzr = pk.loads

    if dsrlzr == pk.loads:
        if not allow_pickle:
            raise ValueError(f"Non-pickle deserializer for type_id {id_} not found and allow_pickle flag was False. Either define a non-pickle deserializer or set the flag to True.")

    return dsrlzr(bytes_)


# Python

# python int
@serializer(int, "__py_int__")
def py_int_serializer(int_):
    return json.dumps(int_).encode()

@deserializer("__py_int__")
def py_int_deserializer(bytes_):
    return json.loads(bytes_.decode())

#python float
@serializer(float, "__py_float__")
def py_float_serializer(float_):
    return json.dumps(float_).encode()

@deserializer("__py_float__")
def py_float_deserializer(bytes_):
    return json.loads(bytes_.decode())

#python complex
@serializer(complex, "__py_complex__")
def py_complex_serializer(complex_):
    tup = (complex_.real, complex_.imag)
    return json.dumps(tup).encode()

@deserializer("__py_complex__")
def py_complex_deserializer(bytes_):
    tup = json.loads(bytes_.decode())
    return complex(*tup)

#python list
@serializer(list, "__py_list__")
def py_list_serializer(list_):
    return json.dumps(list_).encode()

@deserializer("__py_list__")
def py_list_deserializer(bytes_):
    return json.loads(bytes_.decode())

#python tuple
@serializer(tuple, "__py_tuple__")
def py_tuple_serializer(tuple_):
    return json.dumps(tuple_).encode()

@deserializer("__py_tuple__")
def py_tuple_deserializer(bytes_):
    return tuple(json.loads(bytes_.decode()))

#python set
@serializer(set, "__py_set__")
def py_set_serializer(set_):
    return json.dumps(tuple(set_)).encode()

@deserializer("__py_set__")
def py_set_deserializer(bytes_):
    return set(json.loads(bytes_.decode()))

#python dict
@serializer(dict, "__py_dict__")
def py_dict_serializer(dict_):
    return json.dumps(dict_).encode()

@deserializer("__py_dict__")
def py_dict_deserializer(bytes_):
    return json.loads(bytes_.decode())

#python string
@serializer(str, "__py_string__")
def py_string_serializer(string_):
    return string_.encode()

@deserializer("__py_string__")
def py_string_deserializer(bytes_):
    return bytes_.decode()

#python bytes
@serializer(bytes, "__py_bytes__")
def py_bytes_serializer(bytes_):
    return bytes_

@deserializer("__py_bytes__")
def py_bytes_deserializer(bytes_):
    return bytes_

#python bytearray
@serializer(bytearray, "__py_bytearray__")
def py_bytearray_serializer(bytearray_):
    return bytes(bytearray_)

@deserializer("__py_bytearray__")
def py_bytearray_deserializer(bytes_):
    return bytearray(bytes_)

# Numpy

# ndarray
@serializer(np.ndarray, "__numpy_ndarray__")
def numpy_ndarray_serializer(array):
    bytes_io = io.BytesIO(b"")
    np.save(bytes_io, array, allow_pickle=False)
    bytes_io.seek(0)
    return bytes_io.read()
@deserializer("__numpy_ndarray__")
def numpy_ndarray_deserializer(bytes_):
    bytes_io = io.BytesIO(bytes_)
    return np.load(bytes_io)


# Tapr

# NULL
@serializer(NULL, "__tapr_null__")
def tapr_null_serializer(null):
    return json.dumps(0).encode()

@deserializer("__tapr_null__")
def tapr_null_deserializer(bytes_):
    return NULL()


# StandardEngine
@serializer(StandardEngine, "__tapr_standard_engine__")
def tapr_standard_engine_serializer(standard_engine):
    return json.dumps(0).encode()

@deserializer("__tapr_standard_engine__")
def tapr_standard_engine_deserializer(bytes_):
    return StandardEngine()

# ThreadEngine
@serializer(ThreadEngine, "__tapr_thread_engine__")
def tapr_thread_engine_serializer(thread_engine):
    return json.dumps(thread_engine.threads).encode()

@deserializer("__tapr_thread_engine__")
def tapr_standard_engine_deserializer(bytes_):
    threads = json.loads(bytes_.decode())
    return ThreadEngine(threads)

# ProcessEngine
@serializer(ProcessEngine, "__tapr_process_engine__")
def tapr_process_engine_serializer(process_engine):
    return json.dumps(process_engine.processes).encode()

@deserializer("__tapr_process_engine__")
def tapr_standard_engine_deserializer(bytes_):
    processes = json.loads(bytes_.decode())
    return ProcessEngine(processes)


# Plotly

# Figure
@serializer(pl.graph_objs.Figure, "__plotly_figure__")
def plotly_figure_serializer(figure):
    return figure.to_json().encode()

@deserializer("__plotly_figure__")
def plotly_figure_deserializer(bytes_):
    json_ = bytes_.decode()
    dict_ = json.loads(json_)
    return pl.graph_objs.Figure(dict_)
