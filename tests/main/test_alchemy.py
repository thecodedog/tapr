import unittest

from tapr.main.conversion import ntable
from tapr.main.alchemy import NTableMapAlchemy, NTableAlchemy
from tests.testing_utils import assert_ntable_equivalent


class TestNTableAlchemy(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )

    def test_getitem(self):
        result = self._ntbl_a.dim0.alchemy[r"r(.*)1"]
        expected = ntable(
            {
                "r__ALCHEMY__1": {"col1": "r1c1", "col2": "r1c2"},
            }
        )
        assert_ntable_equivalent(result, expected)

    def test_setitem(self):
        self._ntbl_a.dim0.alchemy["3"] = self._ntbl_a.dim0.alchemy[r"r.*(1)"]
        expected = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
                "row3": {"col1": "r1c1", "col2": "r1c2"},
            }
        )
        assert_ntable_equivalent(self._ntbl_a, expected)


if __name__ == "__main__":
    unittest.main()
