import json
import subprocess
import sys

from beqcritic.audit import build_selection_audit
from beqcritic.select import select_from_score_matrix


def test_build_selection_audit_records_clusters_edges_and_labels():
    candidates = ["theorem a : True", "theorem b : True", "theorem c : False"]
    norm = list(candidates)
    labels = [1, 1, 0]
    scores = [
        [1.0, 0.95, 0.2],
        [0.95, 1.0, 0.1],
        [0.2, 0.1, 1.0],
    ]
    res = select_from_score_matrix(
        candidates=candidates,
        norm=norm,
        scores=scores,
        threshold=0.5,
        tie_break="medoid",
        component_rank="size_then_cohesion",
    )

    audit = build_selection_audit(
        problem_id="p1",
        candidates=candidates,
        norm=norm,
        scores=scores,
        selected_index=res.chosen_index,
        selection_method="critic",
        select_mode="cluster",
        threshold=0.5,
        tie_break="medoid",
        component_rank="size_then_cohesion",
        mutual_top_k=0,
        triangle_prune_margin=0.0,
        triangle_prune_keep_best_edge=True,
        cluster_mode="components",
        support_frac=0.7,
        selection_result=res,
        labels=labels,
        top_components=2,
        top_edges=2,
    )

    assert audit["problem_id"] == "p1"
    assert audit["selection"]["chosen_index"] == 0
    assert audit["selection"]["chosen_label"] == 1
    assert audit["graph_stats"]["edges_after"] == 1
    assert audit["components"][0]["indices"] == [0, 1]
    assert audit["components"][0]["n_labeled_correct"] == 2
    assert audit["top_edges"][0] == {"i": 0, "j": 1, "score": 0.95, "label_i": 1, "label_j": 1}


def test_score_and_select_can_emit_audit_jsonl(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    selections = tmp_path / "selections.jsonl"
    audit_path = tmp_path / "audit.jsonl"
    record = {
        "problem_id": "p1",
        "candidates": [
            "theorem a : True := by trivial",
            "theorem b : True := by trivial",
            "theorem c : False := by exact False.elim h",
        ],
        "labels": [1, 1, 0],
    }
    candidates.write_text(json.dumps(record) + "\n", encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.score_and_select",
            "--similarity",
            "bleu",
            "--input",
            str(candidates),
            "--output",
            str(selections),
            "--audit-output",
            str(audit_path),
            "--threshold",
            "0.1",
            "--audit-top-edges",
            "3",
        ],
        check=True,
    )

    selection = json.loads(selections.read_text(encoding="utf-8").splitlines()[0])
    audit = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[0])
    assert audit["problem_id"] == selection["problem_id"]
    assert audit["selection"]["chosen_index"] == selection["chosen_index"]
    assert audit["selection"]["selection_method"] == selection["selection_method"]
    assert audit["candidates"][0]["label"] == 1
    assert audit["top_edges"]
