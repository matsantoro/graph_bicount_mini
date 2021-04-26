import numpy as np
from pathlib import Path
import unittest

from graph_count import biedge_counts_per_dimension, default_temporary_name


class TestRepeatCounts(unittest.TestCase):
    def test_1_biedge_in_simplex(self):
        simplex3 = np.triu(np.ones((4, 4), dtype=int)) - np.diag(np.ones(4, dtype=int))
        simplex3[3, 2] = 1
        self.assertEqual(
            biedge_counts_per_dimension(simplex3),
            {1: 2, 2: 4, 3: 2}
        )

        simplex4 = np.triu(np.ones((5, 5), dtype=int)) - np.diag(np.ones(5, dtype=int))
        simplex4[4, 3] = 1
        self.assertEqual(
            biedge_counts_per_dimension(simplex4),
            {1: 2, 2: 6, 3: 6, 4: 2}
        )

    def test_2_biedge_in_simplex(self):
        simplex3 = np.triu(np.ones((4, 4), dtype=int)) - np.diag(np.ones(4, dtype=int))
        simplex3[3, 2] = 1
        simplex3[2, 1] = 1
        self.assertEqual(
            biedge_counts_per_dimension(simplex3),
            {1: 4, 2: 10, 3: 6}
        )


class TestNoRepeatCounts(unittest.TestCase):
    def test_1_biedge_in_simplex(self):
        simplex3 = np.triu(np.ones((4, 4), dtype=int)) - np.diag(np.ones(4, dtype=int))
        simplex3[3, 2] = 1
        self.assertEqual(
            biedge_counts_per_dimension(simplex3, repeats=False),
            {3: 1}
        )

    def test_2_biedge_in_simplex(self):
        simplex3 = np.triu(np.ones((4, 4), dtype=int)) - np.diag(np.ones(4, dtype=int))
        simplex3[3, 2] = 1
        simplex3[2, 1] = 1
        self.assertEqual(
            biedge_counts_per_dimension(simplex3, repeats=False),
            {3: 2}
        )


class TestBooleanMatrix(unittest.TestCase):
    def test_bool_int(self):
        simplex3int = np.ones((4, 4), dtype=int) - np.diag(np.ones(4, dtype=int))
        simplex3bool = np.multiply(
            np.ones((4, 4), dtype=bool),
            (np.logical_not(np.diag(np.ones(4, dtype=bool))))
        )
        self.assertEqual(
            biedge_counts_per_dimension(simplex3int),
            biedge_counts_per_dimension(simplex3bool)
        )


class TestDoesNotOverride(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Path(default_temporary_name + "1.binary").touch()

    @classmethod
    def tearDownClass(cls):
        Path(default_temporary_name + "1.binary").unlink()

    def test_no_override(self):
        self.assertRaises(FileExistsError, biedge_counts_per_dimension, np.ones((4, 4)))


class TestDoesNotLeaveFiles(unittest.TestCase):
    def test_no_added_files(self):
        initial_file_list = list(Path("").glob("**"))
        biedge_counts_per_dimension(np.ones((4, 4)))
        final_file_list = list(Path("").glob("**"))
        self.assertEqual(initial_file_list, final_file_list)


if __name__ == '__main__':
    unittest.main()
