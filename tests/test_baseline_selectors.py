import json

import pytest

from beqcritic import random_select, self_bleu_select
from beqcritic.schema import SchemaError


def test_self_bleu_select_writes_topk_selection(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "selfbleu.jsonl"
    _write_jsonl(
        candidates,
        [
            {
                "problem_id": "p1",
                "candidates": [
                    "theorem a : True := by trivial",
                    "theorem b : True := by trivial",
                    "theorem c : False := by sorry",
                ],
            }
        ],
    )

    self_bleu_select.main(
        [
            "--input",
            str(candidates),
            "--output",
            str(output),
            "--top-k",
            "2",
            "--emit-topk-text",
        ]
    )

    row = json.loads(output.read_text(encoding="utf-8").strip())
    assert row["problem_id"] == "p1"
    assert row["selection_method"] == "bleu_medoid"
    assert len(row["chosen_indices"]) == 2
    assert len(row["chosen_topk"]) == 2


@pytest.mark.parametrize(
    "selector,args",
    [
        (self_bleu_select.main, []),
        (random_select.main, ["--seed", "7"]),
    ],
    ids=["self_bleu", "random"],
)
def test_baseline_selectors_reject_duplicate_problem_ids_before_writing(
    selector,
    args,
    tmp_path,
):
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "selection.jsonl"
    _write_jsonl(
        candidates,
        [
            {"problem_id": "p1", "candidates": ["theorem a : True := by trivial"]},
            {"problem_id": "p1", "candidates": ["theorem b : True := by trivial"]},
        ],
    )

    with pytest.raises(SchemaError) as excinfo:
        selector(["--input", str(candidates), "--output", str(output), *args])

    message = str(excinfo.value)
    assert f"{candidates}:2" in message
    assert "duplicate problem_id 'p1'" in message
    assert not output.exists()


@pytest.mark.parametrize(
    "selector,args",
    [
        (self_bleu_select.main, []),
        (random_select.main, ["--seed", "7"]),
    ],
    ids=["self_bleu", "random"],
)
def test_baseline_selectors_reject_bad_schema_before_writing(selector, args, tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "selection.jsonl"
    _write_jsonl(candidates, [{"problem_id": "p1", "candidates": ["ok", 3]}])

    with pytest.raises(SchemaError) as excinfo:
        selector(["--input", str(candidates), "--output", str(output), *args])

    message = str(excinfo.value)
    assert f"{candidates}:1" in message
    assert "candidates[1] must be a string" in message
    assert not output.exists()


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
