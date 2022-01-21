import unittest


from tapr.main.conversion import ntable
from tapr.main.engines import StandardEngine, ThreadEngine, ProcessEngine


def func(a, b):
    return a + b


class TestStandardEngine(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )
        self._ntbl_b = ntable(
            {
                "row1": {"col1": "a", "col2": "b"},
                "row2": {"col1": "c", "col2": "d"},
            }
        )

    def test_call(self):
        engine = StandardEngine()
        result = engine(func, self._ntbl_a.reflist, self._ntbl_b.reflist)
        self.assertListEqual(
            result,
            [
                item1 + item2
                for item1, item2 in zip(
                    self._ntbl_a.reflist, self._ntbl_b.reflist
                )
            ],
        )


class TestThreadEngine(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )
        self._ntbl_b = ntable(
            {
                "row1": {"col1": "a", "col2": "b"},
                "row2": {"col1": "c", "col2": "d"},
            }
        )

    def test_call(self):
        engine = ThreadEngine(threads=8)
        result = engine(func, self._ntbl_a.reflist, self._ntbl_b.reflist)
        self.assertListEqual(
            result,
            [
                item1 + item2
                for item1, item2 in zip(
                    self._ntbl_a.reflist, self._ntbl_b.reflist
                )
            ],
        )


class TestProcessEngine(unittest.TestCase):
    def setUp(self):
        self._ntbl_a = ntable(
            {
                "row1": {"col1": "r1c1", "col2": "r1c2"},
                "row2": {"col1": "r2c1", "col2": "r2c2"},
            }
        )
        self._ntbl_b = ntable(
            {
                "row1": {"col1": "a", "col2": "b"},
                "row2": {"col1": "c", "col2": "d"},
            }
        )

    def test_call(self):
        engine = ProcessEngine(processes=8)
        result = engine(func, self._ntbl_a.reflist, self._ntbl_b.reflist)
        self.assertListEqual(
            result,
            [
                item1 + item2
                for item1, item2 in zip(
                    self._ntbl_a.reflist, self._ntbl_b.reflist
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
