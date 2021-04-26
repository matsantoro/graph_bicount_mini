import unittest
import numpy as np
from pathlib import Path
from graph_count import biedge_count_per_dimension

from graph_count import temporary_file_name


class TestRepeatCounts(unittest.TestCase):
    def test_1_biedge_in_simplex(self):
        simplex3 = np.ones((4, 4), dtype=int) - np.diag(np.ones(4, dtype=int))
        simplex3[3, 2] = 1
        self.assertEqual(
            biedge_count_per_dimension(simplex3),
            {1: 2, 2: 4, 3: 2}
        )

        simplex4 = np.ones((4, 4), dtype=int) - np.diag(np.ones(4, dtype=int))
        simplex4[4, 3] = 1
        self.assertEqual(
            biedge_count_per_dimension(simplex4),
            {1: 2, 2: 6, 3: 4, 4: 2}
        )

    def test_2_biedge_in_simplex(self):
        simplex3 = np.ones((4, 4), dtype=int) - np.diag(np.ones(4, dtype=int))
        simplex3[3, 2] = 1
        simplex3[2, 1] = 1
        self.assertEqual(
            biedge_count_per_dimension(simplex3),
            {1: 4, 2: 10, 3: 6}
        )


class TestNoRepeatCounts(unittest.TestCase):
    def test_1_biedge_in_simplex(self):
        simplex3 = np.ones((4, 4), dtype=int) - np.diag(np.ones(4, dtype=int))
        simplex3[3, 2] = 1
        self.assertEqual(
            biedge_count_per_dimension(simplex3, repeats=False),
            {3: 1}
        )

    def test_2_biedge_in_simplex(self):
        simplex3 = np.ones((4, 4), dtype=int) - np.diag(np.ones(4, dtype=int))
        simplex3[3, 2] = 1
        simplex3[2, 1] = 1
        self.assertEqual(
            biedge_count_per_dimension(simplex3, repeats=False),
            {1: 4, 2: 10, 3: 6}
        )


class TestBooleanMatrix(unittest.TestCase):
    def test_bool_int(self):
        simplex3int = np.ones((4, 4), dtype=int) - np.diag(np.ones(4, dtype=int))
        simplex3bool = np.ones((4, 4), dtype=bool).multiply(np.logical_not(np.diag(np.ones(4, dtype=bool))))
        self.assertEqual(
            biedge_count_per_dimension(simplex3int),
            biedge_count_per_dimension(simplex3bool)
        )


class TestDoesNotOverride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Path(temporary_file_name + "1.binary").touch()

    @classmethod
    def tearDownClass(cls):
        Path(temporary_file_name + "1.binary").unlink()

    def test_no_override(self):
        self.assertRaises(FileExistsError, biedge_count_per_dimension, np.ones((4, 4)))


if __name__ == '__main__':
    unittest.main()
