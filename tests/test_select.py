import unittest

from beqcritic.select import global_medoid_index, select_from_score_matrix


class TestSelect(unittest.TestCase):
    def test_global_medoid_index_prefers_hub(self):
        norm = ["a", "bb", "ccc"]
        scores = [
            [1.0, 0.9, 0.9],
            [0.9, 1.0, 0.1],
            [0.9, 0.1, 1.0],
        ]
        idx, cent = global_medoid_index(norm=norm, scores=scores)
        self.assertEqual(idx, 0)
        self.assertGreater(cent, 0.5)

    def test_cluster_selection_picks_largest_component(self):
        candidates = ["a", "bb", "ccc"]
        norm = ["a", "bb", "ccc"]
        scores = [
            [1.0, 0.9, 0.1],
            [0.9, 1.0, 0.1],
            [0.1, 0.1, 1.0],
        ]
        res = select_from_score_matrix(
            candidates=candidates,
            norm=norm,
            scores=scores,
            threshold=0.5,
            tie_break="shortest",
            component_rank="size_then_cohesion",
            mutual_top_k=0,
            triangle_prune_margin=0.0,
            cluster_mode="components",
            support_frac=0.7,
            medoid_simple_top_k=0,
            medoid_simple_max_drop=-1.0,
            simple_weight_chars=1.0,
            simple_weight_binders=0.5,
            simple_weight_prop_assumptions=0.25,
            simple_chars_scale=100.0,
        )
        # The {0,1} component should be chosen; tie_break=shortest picks index 0.
        self.assertEqual(res.component_size, 2)
        self.assertEqual(sorted(res.component_indices), [0, 1])
        self.assertEqual(res.chosen_index, 0)


if __name__ == "__main__":
    unittest.main()

