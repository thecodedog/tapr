import unittest

import numpy as np
import pandas as pd
import xarray as xr

from tapr.main.conversion import ntable, tabulate
from tapr.main.ntable import NTable
from tests.testing_utils import assert_ntable_equivalent


class TestNTable(unittest.TestCase):
    def setUp(self):
        self._dictionary = {
            "x1": {"y1": 0, "y2": "1", "y3": 2},
            "x2": {"y1": "three", "y2": 4, "y3": "5"},
        }

        self._dataframe = pd.DataFrame(self._dictionary)
        self._dataarray = xr.DataArray(
            self._dataframe.to_numpy().T,
            coords={"dim0": ["x1", "x2"], "dim1": ["y1", "y2", "y3"]},
            dims=["dim0", "dim1"],
        )
        reflist = [0, "1", 2, "three", 4, "5"]
        refmap = xr.DataArray(
            np.arange(6).reshape((2, 3)),
            coords={"dim0": ["x1", "x2"], "dim1": ["y1", "y2", "y3"]},
            dims=["dim0", "dim1"],
        )
        self._expected_ntbl = NTable(reflist, refmap)

    def test_ntable_dictionary(self):
        ntbl = ntable(self._dictionary)
        assert_ntable_equivalent(self._expected_ntbl, ntbl)

    def test_ntable_dataframe(self):
        ntbl = ntable(self._dataframe, dims=("dim1", "dim0")).struct.T
        assert_ntable_equivalent(self._expected_ntbl, ntbl)

    def test_ntable_dataarray(self):
        ntbl = ntable(self._dataarray)
        assert_ntable_equivalent(self._expected_ntbl, ntbl)

    def test_ntable_ntable(self):
        ntbl = ntable(self._expected_ntbl)
        assert_ntable_equivalent(self._expected_ntbl, ntbl)

    def test_ntable_collection(self):
        ntbl = ntable((self._expected_ntbl, 3))
        reflist = [(0, 3), ("1", 3), (2, 3), ("three", 3), (4, 3), ("5", 3)]
        refmap = xr.DataArray(
            np.arange(6).reshape((2, 3)),
            coords={"dim0": ["x1", "x2"], "dim1": ["y1", "y2", "y3"]},
            dims=["dim0", "dim1"],
        )
        expected_ntbl = NTable(reflist, refmap)
        assert_ntable_equivalent(expected_ntbl, ntbl)

    def test_ntable_unable(self):
        result = False
        test = 30
        try:
            blah = ntable(test)
        except TypeError:
            result = True
        self.assertTrue(result)


class TestTabulate(unittest.TestCase):
    def setUp(self):
        reflist = [0, "1", 2, "three", 4, "5"]
        self._refmap = xr.DataArray(
            np.arange(6).reshape((2, 3)),
            coords={"dim0": ["x1", "x2"], "dim1": ["y1", "y2", "y3"]},
            dims=["dim0", "dim1"],
        )
        self._ntbl = NTable(reflist, self._refmap)
        self._tuple = (self._ntbl, (3, self._ntbl))
        self._list = [self._ntbl, [3, self._ntbl]]
        self._dictionary = {
            "key1": self._ntbl,
            "key2": {"nested_key1": 3, "nested_key2": self._ntbl},
        }

    def test_tabulate_tuple(self):
        ntbl = tabulate(self._tuple)
        reflist = [
            (0, (3, 0)),
            ("1", (3, "1")),
            (2, (3, 2)),
            ("three", (3, "three")),
            (4, (3, 4)),
            ("5", (3, "5")),
        ]
        expected_ntbl = NTable(reflist, self._refmap)
        assert_ntable_equivalent(ntbl, expected_ntbl)

    def test_tabulate_list(self):
        ntbl = tabulate(self._list)
        reflist = [
            [0, [3, 0]],
            ["1", [3, "1"]],
            [2, [3, 2]],
            ["three", [3, "three"]],
            [4, [3, 4]],
            ["5", [3, "5"]],
        ]
        expected_ntbl = NTable(reflist, self._refmap)
        assert_ntable_equivalent(ntbl, expected_ntbl)

    def test_tabulate_dict(self):
        ntbl = tabulate(self._dictionary)
        reflist = [
            {"key1": 0, "key2": {"nested_key1": 3, "nested_key2": 0}},
            {"key1": "1", "key2": {"nested_key1": 3, "nested_key2": "1"}},
            {"key1": 2, "key2": {"nested_key1": 3, "nested_key2": 2}},
            {
                "key1": "three",
                "key2": {"nested_key1": 3, "nested_key2": "three"},
            },
            {"key1": 4, "key2": {"nested_key1": 3, "nested_key2": 4}},
            {"key1": "5", "key2": {"nested_key1": 3, "nested_key2": "5"}},
        ]
        expected_ntbl = NTable(reflist, self._refmap)
        assert_ntable_equivalent(ntbl, expected_ntbl)


if __name__ == "__main__":
    unittest.main()
