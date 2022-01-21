import unittest

import numpy as np
import pandas as pd
import xarray as xr


from tapr.main.conversion import ntable
from tapr.main.ntable import NTable
from tests.testing_utils import assert_ntable_equivalent


def func(a, b):
    return a + b


class TestNTableStructure(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )
        self._ntbl_a_transpose = NTable(
            self._ntbl_a.reflist, self._ntbl_a.refmap.T
        )

    def test_flat(self):
        result = list(self._ntbl_a.struct.flat)
        expected = ["r1c1", "r1c2", "r2c1", "r2c2"]
        self.assertListEqual(result, expected)

    def test_T(self):
        result = self._ntbl_a.struct.T
        expected = self._ntbl_a_transpose
        assert_ntable_equivalent(result, expected)

    def test_getitem(self):
        # single index in boundaries
        result = self._ntbl_a.struct[0]
        expected = NTable(
            ["r1c1", "r1c2", "r2c1", "r2c2"], self._ntbl_a.refmap[0]
        )
        assert_ntable_equivalent(result, expected)

        # single index out of boundaries
        try:
            result = self._ntbl_a.struct[2]
        except IndexError:
            pass
        except Exception as e:
            self.fail(f"Unexpected exception {e} was raised")
        else:
            self.fail("Expected exception not raised")

        # dual index in boundaries
        result = self._ntbl_a.struct[0, 1]
        expected = NTable(
            ["r1c1", "r1c2", "r2c1", "r2c2"], self._ntbl_a.refmap[0, 1]
        )
        assert_ntable_equivalent(result, expected)

        # dual index out of boundaries
        try:
            result = self._ntbl_a.struct[2, 1]
        except IndexError:
            pass
        except Exception as e:
            self.fail(f"Unexpected exception {e} was raised")
        else:
            self.fail("Expected exception not raised")

    def test_setitem_scalar_value(self):
        # setitem with scalar value
        self._ntbl_a.struct[0] = "test"
        expected = NTable(
            ["test", "test", "r2c1", "r2c2"], self._ntbl_a.refmap
        )
        assert_ntable_equivalent(self._ntbl_a, expected)

    def test_setitem_ntable_value(self):
        # setitem with ntable value
        self._ntbl_a.struct[0] = self._ntbl_a.struct[1]
        expected = NTable(
            ["r2c1", "r2c2", "r2c1", "r2c2"], self._ntbl_a.refmap
        )
        assert_ntable_equivalent(self._ntbl_a, expected)


if __name__ == "__main__":
    unittest.main()
