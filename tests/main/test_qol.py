import unittest

from tapr.main.utils import NULL
from tapr.main.qol import blank, sblank, cartograph
from tapr.main.conversion import ntable

from tests.testing_utils import assert_ntable_equivalent



class TestQol(unittest.TestCase):
    def test_blank(self):
        expected = ntable({"row0":{"col0":NULL(),"col1":NULL()}, "row1":{"col0":NULL(),"col1":NULL()}})
        result = blank(expected.struct.coords, expected.struct.dims)
        assert_ntable_equivalent(result, expected)

    def test_sblank(self):
        expected = ntable({"row0":{"col0":NULL(),"col1":NULL()}, "row1":{"col0":NULL(),"col1":NULL()}})
        # coords = {"rows":[f"row{i}" for i in range(2)], "cols":[f"col{i}" for i in range(2)]
        result = sblank(*expected.struct.shape)
        assert_ntable_equivalent(result, expected)

    def test_cartograph(self):
        expected = ntable({"row0":{"col0":("row0","col0"),"col1":("row0","col1")}, "row1":{"col0":("row1","col0"),"col1":("row1","col1")}})
        result = cartograph(expected)
        assert_ntable_equivalent(result, expected)





if __name__ == "__main__":
    unittest.main()
