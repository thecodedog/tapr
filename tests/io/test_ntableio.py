import unittest

import numpy as np

from tapr.main.conversion import ntable
from tapr.main.engines import ProcessEngine, ThreadEngine
from tapr.io.ntableio import save_ntable, load_ntable
from tests.testing_utils import assert_ntable_equivalent


class Test(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": 3.0, "col2": 3, "col3": (3,), "col4":set([3,]), "col5": complex(3,3)},
                "row2": {"col1": np.array(3), "col2": "3", "col3":[3,], "col4":b"3", "col5":bytearray(b"3")},
            }
        )
        self._ntbl_b = ntable(
            {
                "row1": {"col1": 3.0, "col2": 3, "col3": (3,), "col4":set([3,]), "col5": complex(3,3)},
                "row2": {"col1": np.array(3), "col2": "3", "col3":[3,], "col4":b"3", "col5":bytearray(b"3")},
            }, engine=ThreadEngine(8)
        )
        self._ntbl_c = ntable(
            {
                "row1": {"col1": 3.0, "col2": 3, "col3": (3,), "col4":set([3,]), "col5": complex(3,3)},
                "row2": {"col1": np.array(3), "col2": "3", "col3":[3,], "col4":b"3", "col5":bytearray(b"3")},
            }, engine=ProcessEngine(8)
        )

    def test_save_and_load(self):
        save_ntable(self._ntbl_a, "/tmp/test_a.ntbl")
        result = load_ntable("/tmp/test_a.ntbl")
        assert_ntable_equivalent(result, self._ntbl_a)

        # thread engine
        save_ntable(self._ntbl_b, "/tmp/test_b.ntbl")
        result = load_ntable("/tmp/test_b.ntbl")
        assert_ntable_equivalent(result, self._ntbl_b)

        # process engine
        save_ntable(self._ntbl_c, "/tmp/test_c.ntbl")
        result = load_ntable("/tmp/test_c.ntbl")
        assert_ntable_equivalent(result, self._ntbl_c)







if __name__ == "__main__":
    unittest.main()