import unittest


from tapr.main.conversion import ntable
from tapr.main.tabularization import tabularize
from tapr.main.utils import NULL
from tests.testing_utils import assert_ntable_equivalent


class TestTabularize(unittest.TestCase):
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

    def test_tabularize_good(self):
        def func(a, b):
            return a + b

        dfunc = tabularize()(func)
        result = dfunc(self._ntbl_a, self._ntbl_b)
        expected_result = ntable(
            {
                "row1": {"col1": "r1c1c1", "col2": "r1c2c2"},
                "row2": {"col1": "r2c1c1", "col2": "r2c2c2"},
            }
        )
        assert_ntable_equivalent(result, expected_result)

    def test_tabularize_bad(self):
        def func(a, b):
            return a + b

        dfunc = tabularize()(func)
        result = dfunc(self._ntbl_a, self._ntbl_c)
        expected_result = ntable(
            {
                "row1": {"col1": "r1c1r1c1", "col2": "r1c2r1c2"},
                "row2": {"col1": "r2c1r2c1", "col2": "r2c2r2c2"},
                "row3": {"col1": NULL(), "col2": NULL()},
            }
        )
        assert_ntable_equivalent(result, expected_result)


if __name__ == "__main__":
    unittest.main()
