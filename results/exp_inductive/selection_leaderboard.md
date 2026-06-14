# Selection Leaderboard

- Problems: 55
- Any-correct ceiling: 52.7% (29/55)
- Avg candidates/problem: 9.85
- Best method: `nlverifier`

## Leaderboard

| method | coverage | selected correct | accepted accuracy | correct given any | missed available | abstained |
|---|---:|---:|---:|---:|---:|---:|
| `nlverifier` | 100.0% | 49.1% (27/55) | 49.1% (27/55) | 93.1% (27/29) | 2 | 0 |
| `critic` | 100.0% | 32.7% (18/55) | 32.7% (18/55) | 62.1% (18/29) | 11 | 0 |
| `hybrid` | 100.0% | 32.7% (18/55) | 32.7% (18/55) | 62.1% (18/29) | 11 | 0 |
| `self_bleu` | 100.0% | 30.9% (17/55) | 30.9% (17/55) | 58.6% (17/29) | 12 | 0 |
| `first` | 100.0% | 23.6% (13/55) | 23.6% (13/55) | 44.8% (13/29) | 16 | 0 |
| `shortest` | 100.0% | 10.9% (6/55) | 10.9% (6/55) | 20.7% (6/29) | 23 | 0 |

## Paired Against `nlverifier`

| compared method | both correct | compared only | best only | best lift | discordant | p-value |
|---|---:|---:|---:|---:|---:|---:|
| `critic` | 18 | 0 | 9 | 16.4% | 9 | 0.00390625 |
| `hybrid` | 18 | 0 | 9 | 16.4% | 9 | 0.00390625 |
| `self_bleu` | 17 | 0 | 10 | 18.2% | 10 | 0.00195312 |
| `first` | 13 | 0 | 14 | 25.5% | 14 | 0.00012207 |
| `shortest` | 6 | 0 | 21 | 38.2% | 21 | 9.53674e-07 |

## Pairwise Lift Matrix

Rows are baselines; columns are compared methods. Values are compared minus row accuracy.

| baseline \ compared | `nlverifier` | `critic` | `hybrid` | `self_bleu` | `first` | `shortest` |
|---|---:|---:|---:|---:|---:|---:|
| `nlverifier` | - | -16.4% | -16.4% | -18.2% | -25.5% | -38.2% |
| `critic` | 16.4% | - | 0.0% | -1.8% | -9.1% | -21.8% |
| `hybrid` | 16.4% | 0.0% | - | -1.8% | -9.1% | -21.8% |
| `self_bleu` | 18.2% | 1.8% | 1.8% | - | -7.3% | -20.0% |
| `first` | 25.5% | 9.1% | 9.1% | 7.3% | - | -12.7% |
| `shortest` | 38.2% | 21.8% | 21.8% | 20.0% | 12.7% | - |

## `nlverifier` Wins Over `critic`

| problem_id | best index | baseline index | best candidate | baseline candidate |
|---|---:|---:|---|---|
| Axler\|exercise_6_3 | 8 | 2 | theorem dummy {n : ℕ} {a b : Fin n → ℝ} : (∑ j, a j * b j) ^ 2 ≤ (∑ j, (j + 1) * (a j)... | theorem dummy {n : ℕ} {a : Fin n → ℝ} {b : Fin n → ℝ} : (∑ i, a i * b i) ^ 2 ≤ (∑ i, i... |
| Dummit-Foote\|exercise_4_5_18 | 1 | 0 | theorem dummy {G : Type*} [Group G] [Fintype G] (hG : Fintype.card G = 200) : ∃ P : Syl... | theorem dummy : (Sylow 5 p).Normal := sorry |
| Herstein\|exercise_2_1_26 | 3 | 9 | theorem dummy {G : Type*} [Group G] [Fintype G] (a : G) : ∃ n : ℕ, 0 < n ∧ a ^ n = 1 :=... | theorem dummy (G : Type*) [Group G] [Finite G] (a : G) : ∃ (n : ℕ), a ^ n = 1 := sorry |
| Herstein\|exercise_2_6_15 | 4 | 5 | theorem dummy {G : Type*} [CommGroup G] {m n : ℕ} (hmn : Nat.Coprime m n) (hm : ∃ x : G... | theorem dummy {G : Type*} [Group G] [CommGroup G] (m n : ℕ) (hm : ∃ x, x ^ m = 1) (hn :... |
| Munkres\|exercise_28_6 | 5 | 4 | theorem dummy {X : Type*} [MetricSpace X] [CompactSpace X] (f : X → X) (hf : ∀ x y : X,... | theorem dummy {X : Type u} [MetricSpace X] [CompactSpace X] {f : X → X} (h : ∀ x y : X,... |
| Munkres\|exercise_38_6 | 8 | 4 | theorem dummy {X : Type*} [TopologicalSpace X] [CompletelyRegularSpace X] : ConnectedSp... | theorem dummy {X : Type*} [TopologicalSpace X] [T2Space X] [CompactSpace X] : Connected... |
| Rudin\|exercise_1_13 | 6 | 8 | theorem dummy {x y : ℂ} : \|abs x - abs y\| ≤ abs (x - y) := sorry | theorem dummy (x y : ℂ) : \|\|x\|-\|y\|\| ≤ \|x - y\| := sorry |
| Rudin\|exercise_2_29 | 5 | 0 | theorem dummy (U : Set ℝ) (hU : IsOpen U) : ∃ (S : Set (Set ℝ)), (∀ s ∈ S, ∃ a b : ℝ, s... | theorem dummy {α : Type*} [TopologicalSpace α] (s : Set α) (h : IsOpen s) : ∃ t : Set (... |
| Shakarchi\|exercise_1_13c | 4 | 0 | theorem dummy {Ω : Set ℂ} (hΩ : IsOpen Ω) {f : ℂ → ℂ} (hf : ∀ z ∈ Ω, DifferentiableAt ℂ... | theorem dummy (f : ℂ → ℂ) {s : Set ℂ} (hs : IsOpen s) (hf : AnalyticOn ℂ f s) : (∀ z ∈... |

## `nlverifier` Wins Over `hybrid`

| problem_id | best index | baseline index | best candidate | baseline candidate |
|---|---:|---:|---|---|
| Axler\|exercise_6_3 | 8 | 9 | theorem dummy {n : ℕ} {a b : Fin n → ℝ} : (∑ j, a j * b j) ^ 2 ≤ (∑ j, (j + 1) * (a j)... | theorem dummy {n : ℕ} {a : Fin n → ℝ} {b : Fin n → ℝ} : (∑ i, a i * b i) ^ 2 ≤ (∑ i, i... |
| Dummit-Foote\|exercise_4_5_18 | 1 | 0 | theorem dummy {G : Type*} [Group G] [Fintype G] (hG : Fintype.card G = 200) : ∃ P : Syl... | theorem dummy : (Sylow 5 p).Normal := sorry |
| Herstein\|exercise_2_1_26 | 3 | 2 | theorem dummy {G : Type*} [Group G] [Fintype G] (a : G) : ∃ n : ℕ, 0 < n ∧ a ^ n = 1 :=... | theorem dummy {G : Type*} [Group G] [Finite G] (a : G) : ∃ (n : ℕ), a ^ n = 1 := sorry |
| Herstein\|exercise_2_6_15 | 4 | 5 | theorem dummy {G : Type*} [CommGroup G] {m n : ℕ} (hmn : Nat.Coprime m n) (hm : ∃ x : G... | theorem dummy {G : Type*} [Group G] [CommGroup G] (m n : ℕ) (hm : ∃ x, x ^ m = 1) (hn :... |
| Munkres\|exercise_28_6 | 5 | 4 | theorem dummy {X : Type*} [MetricSpace X] [CompactSpace X] (f : X → X) (hf : ∀ x y : X,... | theorem dummy {X : Type u} [MetricSpace X] [CompactSpace X] {f : X → X} (h : ∀ x y : X,... |
| Munkres\|exercise_38_6 | 8 | 4 | theorem dummy {X : Type*} [TopologicalSpace X] [CompletelyRegularSpace X] : ConnectedSp... | theorem dummy {X : Type*} [TopologicalSpace X] [T2Space X] [CompactSpace X] : Connected... |
| Rudin\|exercise_1_13 | 6 | 0 | theorem dummy {x y : ℂ} : \|abs x - abs y\| ≤ abs (x - y) := sorry | theorem dummy (x y : ℂ) : abs x - abs y ≤ abs (x - y) := sorry |
| Rudin\|exercise_2_29 | 5 | 0 | theorem dummy (U : Set ℝ) (hU : IsOpen U) : ∃ (S : Set (Set ℝ)), (∀ s ∈ S, ∃ a b : ℝ, s... | theorem dummy {α : Type*} [TopologicalSpace α] (s : Set α) (h : IsOpen s) : ∃ t : Set (... |
| Shakarchi\|exercise_1_13c | 4 | 0 | theorem dummy {Ω : Set ℂ} (hΩ : IsOpen Ω) {f : ℂ → ℂ} (hf : ∀ z ∈ Ω, DifferentiableAt ℂ... | theorem dummy (f : ℂ → ℂ) {s : Set ℂ} (hs : IsOpen s) (hf : AnalyticOn ℂ f s) : (∀ z ∈... |

## `nlverifier` Wins Over `self_bleu`

| problem_id | best index | baseline index | best candidate | baseline candidate |
|---|---:|---:|---|---|
| Axler\|exercise_4_4 | 0 | 8 | theorem dummy [Field ℂ] {p : Polynomial ℂ} (h : p.degree = m) : p.roots.toFinset.card =... | theorem dummy {p : Polynomial ℂ} (hp : p.degree = m) : (∀ z, ¬ p.IsRoot z) ↔ ∀ z, ¬ p.d... |
| Axler\|exercise_6_3 | 8 | 9 | theorem dummy {n : ℕ} {a b : Fin n → ℝ} : (∑ j, a j * b j) ^ 2 ≤ (∑ j, (j + 1) * (a j)... | theorem dummy {n : ℕ} {a : Fin n → ℝ} {b : Fin n → ℝ} : (∑ i, a i * b i) ^ 2 ≤ (∑ i, i... |
| Herstein\|exercise_2_1_26 | 3 | 5 | theorem dummy {G : Type*} [Group G] [Fintype G] (a : G) : ∃ n : ℕ, 0 < n ∧ a ^ n = 1 :=... | theorem dummy {G : Type*} [Group G] [Finite G] (a : G) : ∃ n : ℕ, a ^ n = 1 := sorry |
| Herstein\|exercise_2_6_15 | 4 | 10 | theorem dummy {G : Type*} [CommGroup G] {m n : ℕ} (hmn : Nat.Coprime m n) (hm : ∃ x : G... | theorem dummy {G : Type*} [CommGroup G] {m n : ℕ} (hm : ∃ x, x ^ m = 1) (hn : ∃ y, y ^... |
| Munkres\|exercise_22_2a | 6 | 2 | theorem dummy [TopologicalSpace X] [TopologicalSpace Y] (p : ContinuousMap X Y) (f : Co... | theorem dummy {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y] (p : X → Y) (f :... |
| Munkres\|exercise_28_6 | 5 | 2 | theorem dummy {X : Type*} [MetricSpace X] [CompactSpace X] (f : X → X) (hf : ∀ x y : X,... | theorem dummy {X : Type*} [MetricSpace X] [CompactSpace X] (f : X → X) (hf : ∀ x y : X,... |
| Munkres\|exercise_38_6 | 8 | 3 | theorem dummy {X : Type*} [TopologicalSpace X] [CompletelyRegularSpace X] : ConnectedSp... | theorem dummy {X : Type*} [TopologicalSpace X] [T3Space X] : ConnectedSpace X ↔ Connect... |
| Putnam\|exercise_2000_a2 | 26 | 15 | theorem dummy : ∃ᶠ n in at_top, ∃ a b c d e f : ℤ, n = a^2 + b^2 ∧ (n + 1) = c^2 + d^2... | theorem dummy (n : ℕ) : ∃ m : ℕ, ∃ a b c : ℤ, a^2 + b^2 = n ∧ b^2 + c^2 = n+1 ∧ c^2 + a... |
| Rudin\|exercise_1_13 | 6 | 0 | theorem dummy {x y : ℂ} : \|abs x - abs y\| ≤ abs (x - y) := sorry | theorem dummy (x y : ℂ) : abs x - abs y ≤ abs (x - y) := sorry |
| Shakarchi\|exercise_1_13c | 4 | 3 | theorem dummy {Ω : Set ℂ} (hΩ : IsOpen Ω) {f : ℂ → ℂ} (hf : ∀ z ∈ Ω, DifferentiableAt ℂ... | theorem dummy {f : ℂ → ℂ} {s : Set ℂ} (hf : AnalyticOn ℂ f s) (h : ∀ z ∈ s, abs (f z) =... |

## `nlverifier` Wins Over `first`

| problem_id | best index | baseline index | best candidate | baseline candidate |
|---|---:|---:|---|---|
| Axler\|exercise_6_3 | 8 | 0 | theorem dummy {n : ℕ} {a b : Fin n → ℝ} : (∑ j, a j * b j) ^ 2 ≤ (∑ j, (j + 1) * (a j)... | theorem dummy {n : ℕ} {a : Fin n → ℝ} {b : Fin n → ℝ} : ∑ (j : Fin n), a j * b j ^ 2 ≤... |
| Dummit-Foote\|exercise_4_5_18 | 1 | 0 | theorem dummy {G : Type*} [Group G] [Fintype G] (hG : Fintype.card G = 200) : ∃ P : Syl... | theorem dummy : (Sylow 5 p).Normal := sorry |
| Dummit-Foote\|exercise_9_4_2a | 6 | 0 | theorem dummy : Irreducible (Polynomial.X^4 - 4*Polynomial.X^3 + 6 : Polynomial ℤ) := s... | theorem dummy {R : Type*} [Ring R] [IsDomain R] (f : R[X]) (h : f = X ^ 4 - 4 * X ^ 3 +... |
| Herstein\|exercise_2_1_26 | 3 | 0 | theorem dummy {G : Type*} [Group G] [Fintype G] (a : G) : ∃ n : ℕ, 0 < n ∧ a ^ n = 1 :=... | theorem dummy {G : Type*} [Group G] [Fintype G] {p : ℕ} {n : ℕ} [hp : Fact (Nat.Prime p... |
| Herstein\|exercise_2_6_15 | 4 | 0 | theorem dummy {G : Type*} [CommGroup G] {m n : ℕ} (hmn : Nat.Coprime m n) (hm : ∃ x : G... | theorem dummy {G : Type _} [Group G] {m n : ℕ} (hm : ∀ g : G, g ^ m = 1) (hn : ∀ g : G,... |
| Munkres\|exercise_22_2a | 6 | 0 | theorem dummy [TopologicalSpace X] [TopologicalSpace Y] (p : ContinuousMap X Y) (f : Co... | theorem dummy {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y] (p : C(X, Y)) (f... |
| Munkres\|exercise_27_4 | 10 | 0 | theorem dummy {X : Type*} [MetricSpace X] [ConnectedSpace X] (h_two_points : ∃ x y : X,... | theorem dummy (X : Type*) [MetricSpace X] [DiscreteTopology X] [Nonempty X] [ConnectedS... |
| Munkres\|exercise_28_6 | 5 | 0 | theorem dummy {X : Type*} [MetricSpace X] [CompactSpace X] (f : X → X) (hf : ∀ x y : X,... | theorem dummy {X : Type*} [MetricSpace X] [CompactSpace X] (f : X → X) (hf : ∀ x y : X,... |
| Munkres\|exercise_33_7 | 4 | 0 | theorem dummy {X : Type*} [TopologicalSpace X] (hLC : LocallyCompactSpace X) (hH : T2Sp... | theorem dummy {X : Type*} [TopologicalSpace X] [LocallyCompactSpace X] [T2Space X] : T3... |
| Munkres\|exercise_38_6 | 8 | 0 | theorem dummy {X : Type*} [TopologicalSpace X] [CompletelyRegularSpace X] : ConnectedSp... | theorem dummy [CompleteRegularSpace X] : IsConnected X ↔ IsConnected (stoneCech X) := s... |
| Putnam\|exercise_2000_a2 | 26 | 0 | theorem dummy : ∃ᶠ n in at_top, ∃ a b c d e f : ℤ, n = a^2 + b^2 ∧ (n + 1) = c^2 + d^2... | theorem dummy (n : ℕ) : ∃ m : ℕ, m > n ∧ ∃ (a b c : ℕ), m = a ^ 2 + b ^ 2 ∧ m + 1 = a ^... |
| Rudin\|exercise_1_13 | 6 | 0 | theorem dummy {x y : ℂ} : \|abs x - abs y\| ≤ abs (x - y) := sorry | theorem dummy (x y : ℂ) : abs x - abs y ≤ abs (x - y) := sorry |
| Rudin\|exercise_2_29 | 5 | 0 | theorem dummy (U : Set ℝ) (hU : IsOpen U) : ∃ (S : Set (Set ℝ)), (∀ s ∈ S, ∃ a b : ℝ, s... | theorem dummy {α : Type*} [TopologicalSpace α] (s : Set α) (h : IsOpen s) : ∃ t : Set (... |
| Shakarchi\|exercise_1_13c | 4 | 0 | theorem dummy {Ω : Set ℂ} (hΩ : IsOpen Ω) {f : ℂ → ℂ} (hf : ∀ z ∈ Ω, DifferentiableAt ℂ... | theorem dummy (f : ℂ → ℂ) {s : Set ℂ} (hs : IsOpen s) (hf : AnalyticOn ℂ f s) : (∀ z ∈... |

## `nlverifier` Wins Over `shortest`

| problem_id | best index | baseline index | best candidate | baseline candidate |
|---|---:|---:|---|---|
| Artin\|exercise_3_2_7 | 2 | 5 | theorem dummy {K L : Type*} [Field K] [Field L] (φ : K →+* L) : Function.Injective φ :=... | theorem dummy {K L : Type*} [Field K] [Field L] (f : K →* L) : Injective f := sorry |
| Axler\|exercise_4_4 | 0 | 2 | theorem dummy [Field ℂ] {p : Polynomial ℂ} (h : p.degree = m) : p.roots.toFinset.card =... | theorem dummy {p : P} (hp : p.degree = m) : (p.hasDistinctRoots ↔ p.roots ∩ p.derivativ... |
| Axler\|exercise_6_3 | 8 | 5 | theorem dummy {n : ℕ} {a b : Fin n → ℝ} : (∑ j, a j * b j) ^ 2 ≤ (∑ j, (j + 1) * (a j)... | theorem dummy {n : ℕ} {a b : Fin n → ℝ} : (∑ j, a j * b j)^2 ≤ (∑ j, j * a j^2) * (∑ j,... |
| Dummit-Foote\|exercise_4_5_18 | 1 | 0 | theorem dummy {G : Type*} [Group G] [Fintype G] (hG : Fintype.card G = 200) : ∃ P : Syl... | theorem dummy : (Sylow 5 p).Normal := sorry |
| Dummit-Foote\|exercise_9_4_2a | 6 | 7 | theorem dummy : Irreducible (Polynomial.X^4 - 4*Polynomial.X^3 + 6 : Polynomial ℤ) := s... | theorem dummy (x : ℤ[X]) : Irreducible (x^4 - 4*x^3 + 6) := sorry |
| Herstein\|exercise_2_6_15 | 4 | 6 | theorem dummy {G : Type*} [CommGroup G] {m n : ℕ} (hmn : Nat.Coprime m n) (hm : ∃ x : G... | theorem dummy {G : Type _} [Group G] {m n : ℕ} (hm : IsOfOrder G m) (hn : IsOfOrder G n... |
| Herstein\|exercise_4_2_5 | 3 | 6 | theorem dummy (R : Type*) [Ring R] (h : ∀ x : R, x^3 = x) : ∀ x y : R, x * y = y * x :=... | theorem dummy (h : ∀ x : R, x ^ 3 = x) : IsCommutative R := sorry |
| Munkres\|exercise_16_4 | 0 | 8 | theorem dummy {X Y : Type*} [TopologicalSpace X] [TopologicalSpace Y] : (∀ U : Set (X ×... | theorem dummy : IsOpenMap (@Prod.fst X Y) := sorry |
| Munkres\|exercise_27_4 | 10 | 7 | theorem dummy {X : Type*} [MetricSpace X] [ConnectedSpace X] (h_two_points : ∃ x y : X,... | theorem dummy : Uncountable (α := α) := sorry |
| Munkres\|exercise_28_6 | 5 | 3 | theorem dummy {X : Type*} [MetricSpace X] [CompactSpace X] (f : X → X) (hf : ∀ x y : X,... | theorem dummy {X : Type*} [MetricSpace X] [CompactSpace X] {f : X → X} (hf : Isometry f... |
| Munkres\|exercise_30_10 | 4 | 6 | theorem dummy {ι : Type*} [Countable ι] {X : ι → Type*} [∀ i, TopologicalSpace (X i)] (... | theorem dummy {ι : Type*} (X : ι → Type*) [∀ i, TopologicalSpace (X i)] [∀ i, Countable... |
| Munkres\|exercise_31_3 | 6 | 7 | theorem dummy (X : Type*) [PartialOrder X] [TopologicalSpace X] [OrderTopology X] : Reg... | theorem dummy {X : Type*} [Preorder X] [TopologicalSpace X] : RegularSpace X := sorry |
| Munkres\|exercise_33_7 | 4 | 0 | theorem dummy {X : Type*} [TopologicalSpace X] (hLC : LocallyCompactSpace X) (hH : T2Sp... | theorem dummy {X : Type*} [TopologicalSpace X] [LocallyCompactSpace X] [T2Space X] : T3... |
| Munkres\|exercise_38_6 | 8 | 2 | theorem dummy {X : Type*} [TopologicalSpace X] [CompletelyRegularSpace X] : ConnectedSp... | theorem dummy : ConnectedSpace X ↔ ConnectedSpace (StoneCech X) := sorry |
| Putnam\|exercise_2000_a2 | 26 | 5 | theorem dummy : ∃ᶠ n in at_top, ∃ a b c d e f : ℤ, n = a^2 + b^2 ∧ (n + 1) = c^2 + d^2... | theorem dummy : ∃ (n : ℕ) (m : ℕ), n^2 + (n+1)^2 = m^2 ∧ (n+2)^2 = m^2 := sorry |
| Putnam\|exercise_2010_a4 | 11 | 15 | theorem dummy (n : ℕ) (hn : 0 < n) : ¬ Prime (10 ^ 10 ^ 10 ^ n + (10 ^ 10 ^ n) + 10 ^ n... | theorem dummy {m n : ℕ} (hmn : m ∣ n) : (10 ^ m - 1) ∣ (10 ^ n - 1) := sorry |
| Rudin\|exercise_1_13 | 6 | 2 | theorem dummy {x y : ℂ} : \|abs x - abs y\| ≤ abs (x - y) := sorry | theorem dummy (x y : ℂ) : \|x\| - \|y\| ≤ \|x - y\| := sorry |
| Rudin\|exercise_2_29 | 5 | 1 | theorem dummy (U : Set ℝ) (hU : IsOpen U) : ∃ (S : Set (Set ℝ)), (∀ s ∈ S, ∃ a b : ℝ, s... | theorem dummy (s : Set ℝ) (hs : IsOpen s) : s = ⋃ (x ∈ s) (y ∈ s), Set.Ioo x y := sorry |
| Rudin\|exercise_3_22 | 17 | 1 | theorem dummy {X : Type*} [MetricSpace X] [CompleteSpace X] [Nonempty X] (G : ℕ → Set X... | theorem dummy [Nonempty X] {G : ℕ → Set X} (hG : ∀ n, IsOpen (G n)) (hG' : ∀ n, Dense (... |
| Rudin\|exercise_4_2a | 9 | 7 | theorem dummy {X Y : Type*} [MetricSpace X] [MetricSpace Y] (f : X → Y) (hf : Continuou... | theorem dummy (f : C(X, Y)) (E : Set X) : f '' closure E ⊆ closure (f '' E) := sorry |
