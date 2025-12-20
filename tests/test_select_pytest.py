from beqcritic.select import select_from_score_matrix


def test_medoid_simple_topk_can_prefer_simpler_candidate():
    candidates = [
        "theorem t (x : Nat) : True := by trivial",
        "theorem t : True := by trivial",
        "theorem t (h : Prop) : True := by trivial",
    ]
    norm = list(candidates)
    scores = [
        [1.0, 0.9, 0.9],
        [0.9, 1.0, 0.8],
        [0.9, 0.8, 1.0],
    ]

    res = select_from_score_matrix(
        candidates=candidates,
        norm=norm,
        scores=scores,
        threshold=0.5,
        tie_break="medoid",
        component_rank="size_then_cohesion",
        mutual_top_k=0,
        triangle_prune_margin=0.0,
        cluster_mode="components",
        support_frac=0.7,
        medoid_simple_top_k=2,
        medoid_simple_max_drop=-1.0,
        simple_weight_chars=1.0,
        simple_weight_binders=1.0,
        simple_weight_prop_assumptions=1.0,
        simple_chars_scale=100.0,
    )

    assert res.component_size == 3
    # Candidate 0 is the true medoid, but candidate 1 is simpler among the top-2 by centrality.
    assert res.chosen_index == 1

