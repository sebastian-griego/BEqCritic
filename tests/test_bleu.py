import unittest

from beqcritic.bleu import bleu_score_matrix, sym_bleu


class TestBleu(unittest.TestCase):
    def test_score_matrix_is_symmetric(self):
        cand = [
            "theorem a : True := by trivial",
            "theorem b : True := by trivial",
            "theorem c : False := by trivial",
        ]
        _norm, mat = bleu_score_matrix(cand)
        self.assertEqual(len(mat), 3)
        for i in range(3):
            self.assertEqual(len(mat[i]), 3)
            self.assertAlmostEqual(mat[i][i], 1.0, places=6)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(mat[i][j], mat[j][i], places=6)


if __name__ == "__main__":
    unittest.main()
