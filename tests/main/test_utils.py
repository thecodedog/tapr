import unittest

import numpy as np

import xarray as xr

from tapr.main.conversion import ntable
from tapr.main.utils import (
    validate_engine,
    validate_ttype,
    validate_ntable_init,
    any_ntables,
    NULL,
    concatenate_ntables,
    default_refmap,
)
from tapr.main.engines import StandardEngine, ProcessEngine, ThreadEngine

from tests.testing_utils import assert_ntable_equivalent


class BadEngine:
    def __init__(self):
        pass

    def __call__(self, func, *args):
        return ["bad result"]


class TestValidationUtils(unittest.TestCase):
    def test_validate_engine_standard(self):
        standard_engine = StandardEngine()
        try:
            validate_engine(standard_engine)
        except ValueError as e:
            self.fail(f"validation failed due to exception:\n{e}")

    def test_validate_engine_thread(self):
        thread_engine = ThreadEngine(8)
        try:
            validate_engine(thread_engine)
        except ValueError as e:
            self.fail(f"validation failed due to exception:\n{e}")

    def test_validate_engine_process(self):
        process_engine = ProcessEngine(8)
        try:
            validate_engine(process_engine)
        except ValueError as e:
            self.fail(f"validation failed due to exception:\n{e}")

    def test_validate_engine_bad(self):
        bad_engine = BadEngine()
        try:
            validate_engine(bad_engine)
        except ValueError:
            pass
        except Exception:
            self.fail("unexpected exception was raised")
        else:
            self.fail("An exception was not raised")

    def test_validate_ttype_good(self):
        good_ttype = {np.ndarray, np.int32, int, float, object}
        try:
            validate_ttype(good_ttype)
        except ValueError as e:
            self.fail(f"validation failed due to exception:\n{e}")

    def test_validate_ttype_bad(self):
        bad_ttype = {np.ndarray, np.int64, int, float, object, "string"}
        try:
            validate_ttype(bad_ttype)
        except ValueError:
            pass
        except Exception:
            self.fail("unexpected exception was raised")
        else:
            self.fail("An exception was not raised")

    def test_validate_ntable_init_good(self):
        reflist = [1, 2, 3, 4, 5, 6]
        refmap = xr.DataArray(
            np.arange(6).reshape(2, 3),
            dims=("dim1", "dim2"),
            coords={"dim1": ["x1", "x2"], "dim2": ["y1", "y2", "y3"]},
        )
        engine = StandardEngine()
        ttype = {str, int, float}
        try:
            validate_ntable_init(reflist, refmap, engine, ttype)
        except Exception as e:
            self.fail(f"validation failed due to exception:\n{e}")

    def test_validate_ntable_init_bad(self):
        # refmap has invalid index for reflist
        reflist1 = [1, 2, 3, 4, 5]
        refmap1 = xr.DataArray(
            np.arange(6).reshape(2, 3),
            dims=("dim1", "dim2"),
            coords={"dim1": ["x1", "x2"], "dim2": ["y1", "y2", "y3"]},
        )
        engine = StandardEngine()
        ttype = {str, int, float}
        try:
            validate_ntable_init(reflist1, refmap1, engine, ttype)
        except ValueError:
            pass
        except Exception:
            self.fail("unexpected exception was raised")
        else:
            self.fail("An exception was not raised")

        # refmap has empty coordinates
        reflist2 = [1, 2, 3, 4, 5, 6]
        refmap2 = xr.DataArray(
            np.arange(6).reshape(2, 3),
        )
        engine = StandardEngine()
        ttype = {str, int, float}
        try:
            validate_ntable_init(reflist2, refmap2, engine, ttype)
        except ValueError:
            pass
        except Exception:
            self.fail("unexpected exception was raised")
        else:
            self.fail("An exception was not raised")

        # reflist is not a list
        reflist3 = (1, 2, 3, 4, 5, 6)
        refmap3 = xr.DataArray(
            np.arange(6).reshape(2, 3),
            dims=("dim1", "dim2"),
            coords={"dim1": ["x1", "x2"], "dim2": ["y1", "y2", "y3"]},
        )
        engine = StandardEngine()
        ttype = {str, int, float}
        try:
            validate_ntable_init(reflist3, refmap3, engine, ttype)
        except TypeError:
            pass
        except Exception:
            self.fail("unexpected exception was raised")
        else:
            self.fail("An exception was not raised")

        # refmap is not is not a DataArray
        reflist4 = [1, 2, 3, 4, 5, 6]
        refmap4 = np.arange(6).reshape(2, 3)
        engine = StandardEngine()
        ttype = {str, int, float}
        try:
            validate_ntable_init(reflist4, refmap4, engine, ttype)
        except TypeError:
            pass
        except Exception:
            self.fail("unexpected exception was raised")
        else:
            self.fail("An exception was not raised")

    def test_str_ntable_element(self):
        ntbl = ntable(
            {
                "x1": {
                    "y1": np.arange(6),
                    "y2": xr.DataArray(np.arange(6)),
                    "y3": 2,
                },
                "x2": {
                    "y1": "three_four_five_six_seven_eight_nine_ten",
                    "y2": [1, 2, 3],
                    "y3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                },
            }
        )
        expected = """dim1                  y1                     y2                y3
dim0                                                             
x1    ndarray,(6,),int64  data array,(6,),int64                 2
x2         three...e_ten              [1, 2, 3]  [1, 2]...[9, 10]
Coordinates:
  * dim0     (dim0) <U2 'x1' 'x2'
  * dim1     (dim1) <U2 'y1' 'y2' 'y3'
Engine:
Standard (serial) Engine
Ttype:
DataArray|int|list|ndarray|str"""
        result = str(ntbl)
        self.assertEqual(result, expected)

    def test_str_ntable(self):
        dictionary = {}
        for i in range(150):
            dictionary[f"row{i}"] = {}
            for j in range(30):
                dictionary[f"row{i}"][f"col{j}"] = {}
                for k in range(15):
                    dictionary[f"row{i}"][f"col{j}"][f"page{k}"] = np.arange(
                        10
                    )
        ntbl = ntable(dictionary)
        # big boi
        result = str(ntbl)
        expected = """page0:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page1:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page2:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page3:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page4:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

. . .

###############################################################################

page10:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page11:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page12:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page13:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page14:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################


Coordinates:
  * dim0     (dim0) <U6 'row0' 'row1' 'row2' ... 'row147' 'row148' 'row149'
  * dim1     (dim1) <U5 'col0' 'col1' 'col2' 'col3' ... 'col27' 'col28' 'col29'
  * dim2     (dim2) <U6 'page0' 'page1' 'page2' ... 'page12' 'page13' 'page14'
Engine:
Standard (serial) Engine
Ttype:
ndarray"""
        self.assertEqual(result, expected)

        # 0 dim ntbl
        result = str(ntbl.struct[0, 0, 0])
        expected = """ndarray,(10,),int64
Coordinates:
    dim0     <U6 'row0'
    dim1     <U5 'col0'
    dim2     <U6 'page0'
Engine:
Standard (serial) Engine
Ttype:
ndarray"""
        self.assertEqual(result, expected)

        # smol boi
        result = str(ntbl.dim2.page0.struct[0:5, 0:3])
        expected = """dim1                 col0                 col1                 col2
dim0                                                               
row0  ndarray,(10,),int64  ndarray,(10,),int64  ndarray,(10,),int64
row1  ndarray,(10,),int64  ndarray,(10,),int64  ndarray,(10,),int64
row2  ndarray,(10,),int64  ndarray,(10,),int64  ndarray,(10,),int64
row3  ndarray,(10,),int64  ndarray,(10,),int64  ndarray,(10,),int64
row4  ndarray,(10,),int64  ndarray,(10,),int64  ndarray,(10,),int64
Coordinates:
  * dim0     (dim0) <U6 'row0' 'row1' 'row2' 'row3' 'row4'
  * dim1     (dim1) <U5 'col0' 'col1' 'col2'
    dim2     <U6 'page0'
Engine:
Standard (serial) Engine
Ttype:
ndarray"""

        self.assertEqual(result, expected)

        # biggish boi
        result = str(ntbl.struct[:, :, 0:5])
        # print(result)
        expected = """page0:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page1:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page2:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page3:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################

page4:

dim1             col0           col1           col2           col3  ...          col26          col27          col28          col29
dim0                                                                ...                                                            
row0    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row1    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row2    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row3    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row4    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row5    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row6    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row7    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row8    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row9    ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row140  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row141  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row142  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row143  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row144  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row145  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row146  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row147  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row148  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64
row149  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64  ...  ndarr...int64  ndarr...int64  ndarr...int64  ndarr...int64

[20 rows x 30 columns]

###############################################################################


Coordinates:
  * dim0     (dim0) <U6 'row0' 'row1' 'row2' ... 'row147' 'row148' 'row149'
  * dim1     (dim1) <U5 'col0' 'col1' 'col2' 'col3' ... 'col27' 'col28' 'col29'
  * dim2     (dim2) <U6 'page0' 'page1' 'page2' 'page3' 'page4'
Engine:
Standard (serial) Engine
Ttype:
ndarray"""

        self.assertEqual(result, expected)

    def test_any_ntables(self):
        ntbl = ntable(
            {
                "x1": {"y1": 0, "y2": "1", "y3": 2},
                "x2": {"y1": "three", "y2": 4, "y3": "5"},
            }
        )
        self.assertTrue(any_ntables((0, 1, 2, ntbl, 0, ntbl)))

    def test_null(self):
        null = NULL()
        self.assertEqual("NULL", str(null))
        self.assertEqual("NULL", repr(null))
        self.assertEqual("NULL", null.__ntable_element__str__())
        self.assertTrue(isinstance(null[0], NULL))
        self.assertTrue(isinstance(null(), NULL))
        self.assertTrue(isinstance(null + 10, NULL))

    def test_concat_ntables(self):
        dictionary = {
            "x1": {"y1": 0, "y2": "1", "y3": 2},
            "x2": {"y1": "three", "y2": 4, "y3": "5"},
        }
        ntbl = ntable(dictionary)
        new_ntbl = concatenate_ntables((ntbl, ntbl), dim="dim2")
        expected_ntbl = ntable(
            {0: dictionary, 1: dictionary}, dims=("dim2", "dim0", "dim1")
        ).struct.transpose("dim0", "dim1", "dim2")

        assert_ntable_equivalent(new_ntbl, expected_ntbl)

        dictionary1 = {
            "x1": {"y1": 0, "y2": "1", "y3": NULL()},
            "x2": {"y1": "three", "y2": 4, "y3": NULL()},
        }

        new_ntbl1 = concatenate_ntables((ntbl, ntbl.struct[:,0:2]), dim="dim2")

        expected_ntbl1 = ntable(
            {0: dictionary, 1: dictionary1}, dims=("dim2", "dim0", "dim1")
        ).struct.transpose("dim0", "dim1", "dim2")

        assert_ntable_equivalent(new_ntbl1, expected_ntbl1)

    def test_default_refmap(self):
        refmap = default_refmap(3,2)
        expected = xr.DataArray(np.arange(6).reshape((3,2)), coords={"dim0":["coord0","coord1","coord2"], "dim1":["coord0","coord1"]})
        self.assertTrue(refmap.equals(expected))


if __name__ == "__main__":
    unittest.main()
