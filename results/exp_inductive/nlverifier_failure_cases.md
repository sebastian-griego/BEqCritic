# NLVerifier inductive failures (id-disjoint test)

Each case has at least one correct candidate, but NLVerifier selected an incorrect one.

## Case 1 — problem_id Herstein|exercise_4_3_25

- chosen_score: 7.0185, best_correct_score: 3.7702
- chosen_len: 77, best_correct_len: 77
- jaccard_nl_overlap: chosen=0.154, best_correct=0.115
- note: Chosen has higher NL lexical overlap than best-correct; possible surface-match bias.

NL statement:
```
Let $R$ be the ring of $2 \times 2$ matrices over the real numbers; suppose that $I$ is an ideal of $R$. Show that $I = (0)$ or $I = R$.
```
Chosen candidate (incorrect):
```
theorem dummy (I : Ideal (Matrix (Fin 2) (Fin 2) R)) : I = ⊥ ∨ I = ⊤ := sorry
```
Best correct candidate:
```
theorem dummy {I : Ideal (Matrix (Fin 2) (Fin 2) ℝ)} : I = ⊥ ∨ I = ⊤ := sorry
```

## Case 2 — problem_id Axler|exercise_1_6

- chosen_score: -3.6997, best_correct_score: -4.3730
- chosen_len: 121, best_correct_len: 253
- jaccard_nl_overlap: chosen=0.059, best_correct=0.081
- note: Best-correct has higher NL lexical overlap; possible reasoning or feature mismatch. Chosen is much shorter than best-correct.

NL statement:
```
Give an example of a nonempty subset $U$ of $\mathbf{R}^2$ such that $U$ is closed under addition and under taking additive inverses (meaning $-u \in U$ whenever $u \in U$), but $U$ is not a subspace of $\mathbf{R}^2$.
```
Chosen candidate (incorrect):
```
theorem dummy : ∃ U : Set ℝ², U.Nonempty ∧ ClosedUnder (· + ·) U ∧ ClosedUnder (fun u => -u) U ∧ ¬IsSubspace ℝ U := sorry
```
Best correct candidate:
```
theorem dummy : ∃ (U : Set (EuclideanSpace ℝ (Fin 2))), U.Nonempty ∧ (∀ (x : EuclideanSpace ℝ (Fin 2)), x ∈ U → -x ∈ U) ∧ (∀ (x y : EuclideanSpace ℝ (Fin 2)), x ∈ U → y ∈ U → x + y ∈ U) ∧ ¬(∃ (s : Submodule ℝ (EuclideanSpace ℝ (Fin 2))), s = U) := sorry
```

