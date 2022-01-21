import unittest

import numpy as np
import pandas as pd
import xarray as xr


from tapr.main.processing import broadcast_tables, tabular_map
from tapr.main.conversion import ntable
from tapr.main.ntable import NTable
from tapr.main.utils import NULL
from tapr.main.engines import StandardEngine
from tests.testing_utils import assert_ntable_equivalent


class TestBroadcastTables(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )
        self._ntbl_b = ntable({"col1": "c1", "col2": "c2"}, dims=("dim1",))
        self._ntbl_c = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
                "row3": {"col1": "r3c1", "col2": "r3c2"},
            }
        )

    def test_broadcast_tables_good(self):
        ntbl1, ntbl2 = broadcast_tables(self._ntbl_a, self._ntbl_b)
        assert_ntable_equivalent(ntbl1, self._ntbl_a)
        reflist = ["c1", "c2", "c1", "c2"]
        test_ntbl2 = NTable(reflist, self._ntbl_a.refmap)
        assert_ntable_equivalent(ntbl2, test_ntbl2)

    def test_broadcast_tables_bad(self):
        ntbl1, ntbl2 = broadcast_tables(self._ntbl_a, self._ntbl_c)
        test_ntbl1 = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
                "row3": {"col1": NULL(), "col2": NULL()},
            }
        )
        assert_ntable_equivalent(ntbl1, test_ntbl1)
        assert_ntable_equivalent(ntbl2, self._ntbl_c)


class TestTabularMap(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )
        self._ntbl_b = ntable({"col1": "c1", "col2": "c2"}, dims=("dim1",))
        # self._ntbl_c = ntable(
        #     {
        #         "row1": {"col1": "r1c1", "col2": "r1c2"},
        #         "row2": {"col1": "r2c1", "col2": "r2c2"},
        #         "row3": {"col1": "r3c1", "col2": "r3c2"},
        #     }
        # )

    def test_tabular_map_good(self):
        def func(a, b):
            return a + b

        result = tabular_map(
            (func, StandardEngine()), self._ntbl_a, self._ntbl_a
        )

        expected_ntbl = ntable(
            {
                "row1": {"col1": "r1c1r1c1", "col2": "r1c2r1c2"},
                "row2": {"col1": "r2c1r2c1", "col2": "r2c2r2c2"},
            }
        )
        assert_ntable_equivalent(result, expected_ntbl)

    def test_tabular_map_bad(self):
        def func(a, b):
            return a + b

        try:
            result = tabular_map(
                (func, StandardEngine()), self._ntbl_a, self._ntbl_b
            )
        except ValueError:
            pass
        except Exception as e:
            self.fail(f"Unexpected exception {e} was raised")
        else:
            self.fail("ExpectedException not raised")


if __name__ == "__main__":
    unittest.main()
