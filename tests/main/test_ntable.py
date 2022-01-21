from collections import namedtuple
import unittest

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
import operator as op

from tapr.main.ntable import NTable
from tapr.main.conversion import ntable
from tapr.main.handling import FunctionError
from tapr.main.alchemy import NTableMapAlchemy, NTableAlchemy
from tests.testing_utils import assert_ntable_equivalent


def aaa():
    return "aaa"


@dataclass
class Test:
    a: int


class TestNTableCreation(unittest.TestCase):
    def setUp(self):
        self._reflist_a = [0, "1", 2, "three", 4, "5"]
        self._refmap_a = xr.DataArray(
            np.arange(6).reshape((2, 3)),
            coords={"dim_0": ["x1", "x2"], "dim_1": ["y1", "y2", "y3"]},
            dims=["dim_0", "dim_1"],
        )
        self._reflist_b = [
            0,
            "1",
            2,
            "three",
            4,
            "5",
            6,
            "sev",
            8.0,
            9,
            10,
            11,
        ]
        self._refmap_b = xr.DataArray(
            np.arange(12).reshape((3, 4)),
            coords={
                "dim_0": ["x1", "x2", "x3"],
                "dim_1": ["y1", "y2", "y3", "y4"],
            },
            dims=["dim_0", "dim_1"],
        )

    def test_init(self):
        # test good init
        ntbl = NTable(self._reflist_a, self._refmap_a)
        # test mismatch reflist/refmap
        self.assertRaises(ValueError, NTable, self._reflist_a, self._refmap_b)


class TestNTableCoversion(unittest.TestCase):
    def setUp(self):
        self._reflist_a = [0, "1", 2, "three", 4, "5"]
        self._refmap_a = xr.DataArray(
            np.arange(6).reshape((2, 3)),
            coords={"dim_0": ["x1", "x2"], "dim_1": ["y1", "y2", "y3"]},
            dims=["dim_0", "dim_1"],
        )
        self._ntable_a = NTable(self._reflist_a, self._refmap_a)

    def test_to_dictionary(self):
        dictionary = self._ntable_a.to_dictionary()
        self.assertDictEqual(
            dictionary,
            {
                "x1": {"y1": 0, "y2": "1", "y3": 2},
                "x2": {"y1": "three", "y2": 4, "y3": "5"},
            },
        )

    def test_to_pandas(self):
        dframe = self._ntable_a.to_pandas()
        test_dframe = pd.DataFrame(
            {
                "x1": {"y1": 0, "y2": "1", "y3": 2},
                "x2": {"y1": "three", "y2": 4, "y3": "5"},
            }
        ).T
        test_dframe.index.rename("dim_0", inplace=True)
        test_dframe.columns.rename("dim_1", inplace=True)
        pd.util.testing.assert_frame_equal(dframe, test_dframe)

    def test_to_data_array(self):
        darray = self._ntable_a.to_data_array()
        test_darray = xr.DataArray(
            np.array([[0, "1", 2], ["three", 4, "5"]], dtype="object"),
            coords={"dim_0": ["x1", "x2"], "dim_1": ["y1", "y2", "y3"]},
            dims=["dim_0", "dim_1"],
        )
        xr.testing.assert_equal(darray, test_darray)


class TestNTableCore(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )

        self._ntbl_b = ntable(
            {
                "row1": {"col1": list("r1c1"), "col2": list("r1c2")},
                "row2": {"col1": list("r2c1"), "col2": list("r2c2")},
            }
        )

        self._ntbl_c = ntable(
            {
                "row1": {"col1": aaa, "col2": aaa},
                "row2": {"col1": aaa, "col2": aaa},
            }
        )

        self._ntbl_d = ntable(
            {
                "row1": {"col1": np.arange(3), "col2": np.arange(3)},
                "row2": {"col1": np.arange(3), "col2": np.arange(3)},
            }
        )

    def test_iter(self):
        expected0 = ntable(
            {
                "row1": {"col1": "r", "col2": "r"},
                "row2": {"col1": "r", "col2": "r"},
            }
        )
        expected1 = ntable(
            {
                "row1": {"col1": "1", "col2": "1"},
                "row2": {"col1": "2", "col2": "2"},
            }
        )
        expected2 = ntable(
            {
                "row1": {"col1": "c", "col2": "c"},
                "row2": {"col1": "c", "col2": "c"},
            }
        )
        expected3 = ntable(
            {
                "row1": {"col1": "1", "col2": "2"},
                "row2": {"col1": "1", "col2": "2"},
            }
        )
        expecteds = [expected0, expected1, expected2, expected3]
        for result, expected in zip(self._ntbl_a, expecteds):
            assert_ntable_equivalent(result, expected)

    def test_getitem(self):
        # in bounds
        expected = ntable(
            {
                "row1": {"col1": "r", "col2": "r"},
                "row2": {"col1": "r", "col2": "r"},
            }
        )
        result = self._ntbl_a[0]
        assert_ntable_equivalent(result, expected)

        # out of bounds
        expected = ntable(
            {
                "row1": {
                    "col1": FunctionError(
                        IndexError("string index out of range"),
                        op.getitem,
                        "r1c1",
                        4,
                    ),
                    "col2": FunctionError(
                        IndexError("string index out of range"),
                        op.getitem,
                        "r1c2",
                        4,
                    ),
                },
                "row2": {
                    "col1": FunctionError(
                        IndexError("string index out of range"),
                        op.getitem,
                        "r2c1",
                        4,
                    ),
                    "col2": FunctionError(
                        IndexError("string index out of range"),
                        op.getitem,
                        "r2c2",
                        4,
                    ),
                },
            }
        )
        result = self._ntbl_a[4]
        assert_ntable_equivalent(result, expected)

    def test_setitem(self):
        self._ntbl_b[1] = "lol"
        expected = ntable(
            {
                "row1": {
                    "col1": ["r", "lol", "c", "1"],
                    "col2": ["r", "lol", "c", "2"],
                },
                "row2": {
                    "col1": ["r", "lol", "c", "1"],
                    "col2": ["r", "lol", "c", "2"],
                },
            }
        )
        assert_ntable_equivalent(self._ntbl_b, expected)

    def test_call(self):
        expected = ntable(
            {
                "row1": {"col1": "aaa", "col2": "aaa"},
                "row2": {"col1": "aaa", "col2": "aaa"},
            }
        )
        result = self._ntbl_c()
        assert_ntable_equivalent(result, expected)

    def test_array_ufunc(self):
        expected = ntable(
            {
                "row1": {
                    "col1": np.sin(np.arange(3)),
                    "col2": np.sin(np.arange(3)),
                },
                "row2": {
                    "col1": np.sin(np.arange(3)),
                    "col2": np.sin(np.arange(3)),
                },
            }
        )
        result = np.sin((self._ntbl_d))
        self.assertTrue(all(np.array_equal(result, expected).struct.flat))

    def test_array_function(self):
        expected = ntable(
            {
                "row1": {
                    "col1": np.vstack((np.arange(3), np.arange(3))),
                    "col2": np.vstack((np.arange(3), np.arange(3))),
                },
                "row2": {
                    "col1": np.vstack((np.arange(3), np.arange(3))),
                    "col2": np.vstack((np.arange(3), np.arange(3))),
                },
            }
        )
        result = np.vstack((self._ntbl_d, self._ntbl_d))
        self.assertTrue(all(np.array_equal(result, expected).struct.flat))

    def test_getattr(self):
        result = self._ntbl_d.shape
        expected = ntable(
            {
                "row1": {"col1": (3,), "col2": (3,)},
                "row2": {"col1": (3,), "col2": (3,)},
            }
        )
        assert_ntable_equivalent(result, expected)

    def test_str(self):
        result = str(self._ntbl_a)
        expected = """dim1    col1    col2
dim0                
row1  "r1c1"  "r1c2"
row2  "r2c1"  "r2c2"
Coordinates:
  * dim0     (dim0) <U4 'row1' 'row2'
  * dim1     (dim1) <U4 'col1' 'col2'
Engine:
Standard (serial) Engine
Ttype:
str"""
        self.assertEqual(result, expected)


class TestNTableMap(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )

    def test_setitem_old_key(self):
        dim0_map = self._ntbl_a.ntable_map("dim0")
        dim0_map["row1"] = dim0_map["row2"]
        expected = ntable(
            {
                "row1": {"col1": "r2c1", "col2": "r2c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )
        assert_ntable_equivalent(self._ntbl_a, expected)

    def test_setitem_new_key(self):
        dim0_map = self._ntbl_a.ntable_map("dim0")
        dim0_map["row3"] = dim0_map["row2"]
        expected = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
                "row3": {"col1": "r2c1", "col2": "r2c2"},
            }
        )
        assert_ntable_equivalent(self._ntbl_a, expected)

    def test_setitem_list_key(self):
        insert = ntable(
            {
                "row1": {"col1": "a", "col2": "b"},
                "row2": {"col1": "c", "col2": "d"},
            }
        )
        dim0_map = self._ntbl_a.ntable_map("dim0")
        dim0_map[["row1", "row2"]] = insert
        expected = ntable(
            {
                "row1": {"col1": "a", "col2": "b"},
                "row2": {"col1": "c", "col2": "d"},
            }
        )
        assert_ntable_equivalent(self._ntbl_a, expected)

    def test_setitem_list_key_invalid_value_length(self):
        insert = ntable(
            {
                "row1": {"col1": "a", "col2": "b"},
                "row2": {"col1": "c", "col2": "d"},
            }
        )
        dim0_map = self._ntbl_a.ntable_map("dim0")
        try:
            dim0_map[["row1"]] = insert
        except ValueError:
            self.assertTrue(True)

    def test_setitem_list_key_scalar_value(self):
        dim0_map = self._ntbl_a.ntable_map("dim0")
        dim0_map[["row3", "row4"]] = 3
        expected = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
                "row3": {"col1": 3, "col2": 3},
                "row4": {"col1": 3, "col2": 3},
            }
        )
        assert_ntable_equivalent(self._ntbl_a, expected)


class TestTabularizedAttributes(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": Test(1), "col2": Test(2)},
                "row2": {"col1": Test(3), "col2": Test(4)},
            }
        )

    def test_getattr(self):
        result = self._ntbl_a.tattr.a
        expected = ntable(
            {
                "row1": {"col1": 1, "col2": 2},
                "row2": {"col1": 3, "col2": 4},
            }
        )
        assert_ntable_equivalent(result, expected)

    def test_setattr(self):
        self._ntbl_a.tattr.a = 10
        expected = ntable(
            {
                "row1": {"col1": Test(10), "col2": Test(10)},
                "row2": {"col1": Test(10), "col2": Test(10)},
            }
        )
        assert_ntable_equivalent(self._ntbl_a, expected)


class TestMisc(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": Test(1), "col2": Test(2)},
                "row2": {"col1": Test(3), "col2": Test(4)},
            }
        )

    def test_ntable_map_ntable(self):
        result = self._ntbl_a.dim0.ntable
        self.assertTrue(result is self._ntbl_a)

    def test_ntable_map_alchemy(self):
        result = self._ntbl_a.dim0.alchemy
        self.assertTrue(isinstance(result, NTableMapAlchemy))

    def test_ntable_map_str(self):
        result = str(self._ntbl_a.dim0)
        expected = """{'row1': dim1
col1    Test
col2    Test
dtype: object
Coordinates:
    dim0     <U4 'row1'
  * dim1     (dim1) <U4 'col1' 'col2'
Engine:
Standard (serial) Engine
Ttype:
Test, 'row2': dim1
col1    Test
col2    Test
dtype: object
Coordinates:
    dim0     <U4 'row2'
  * dim1     (dim1) <U4 'col1' 'col2'
Engine:
Standard (serial) Engine
Ttype:
Test}"""
        self.assertEqual(result, expected)

    def test_ntable_map_repr(self):
        result = repr(self._ntbl_a.dim0)
        expected = """{'row1': dim1
col1    Test
col2    Test
dtype: object
Coordinates:
    dim0     <U4 'row1'
  * dim1     (dim1) <U4 'col1' 'col2'
Engine:
Standard (serial) Engine
Ttype:
Test, 'row2': dim1
col1    Test
col2    Test
dtype: object
Coordinates:
    dim0     <U4 'row2'
  * dim1     (dim1) <U4 'col1' 'col2'
Engine:
Standard (serial) Engine
Ttype:
Test}"""
        self.assertEqual(result, expected)

    def test_ntable_map_delitem(self):
        try:
            del self._ntbl_a.dim0["row1"]
        except NotImplementedError:
            self.assertTrue(True)

    def test_ntable_map_contains(self):
        result = self._ntbl_a.dim0.contains("1")
        expected = ntable(
            {
                "row1": {"col1": Test(1), "col2": Test(2)},
            }
        )
        assert_ntable_equivalent(result, expected)

    def test_ntable_map_matches(self):
        result = self._ntbl_a.dim0.matches(r"r.*2")
        expected = ntable(
            {
                "row2": {"col1": Test(3), "col2": Test(4)},
            }
        )
        assert_ntable_equivalent(result, expected)

    def test_ntable_map_relabel(self):
        result = self._ntbl_a.dim0.relabel(row1="row3", row2="row4")
        expected = ntable(
            {
                "row3": {"col1": Test(1), "col2": Test(2)},
                "row4": {"col1": Test(3), "col2": Test(4)},
            }
        )
        assert_ntable_equivalent(result, expected)

    def test_ntable_alchemy(self):
        result = self._ntbl_a.alchemy
        self.assertTrue(isinstance(result, NTableAlchemy))

    def test_ntable_getattr_err(self):
        result = False
        try:
            self._ntbl_a.spacedoge
        except AttributeError:
            result = True

        self.assertTrue(result)

    def test_ntable_item(self):
        result = self._ntbl_a.struct.loc["row1", "col1"].item()
        expected = Test(1)
        self.assertEqual(result, expected)

    def test_ntable_to_pandas_error(self):
        ntbl = ntable(
            {
                "page1": {
                    "row3": {"col1": Test(1), "col2": Test(2)},
                    "row4": {"col1": Test(3), "col2": Test(4)},
                }
            }
        )
        result = False
        try:
            blah = ntbl.to_pandas()
        except ValueError:
            result = True

        self.assertTrue(result)


if __name__ == "__main__":
    unittest.main()
