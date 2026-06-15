import json

import pytest

from beqcritic.jsonl import JsonlError
from beqcritic.paper_pipeline import beq_plus_eval as bpe
from beqcritic.paper_pipeline import beq_plus_oracle as bpo


def test_beq_plus_selection_loader_rejects_duplicate_problem_ids(tmp_path):
    path = tmp_path / "selections.jsonl"
    path.write_text(
        json.dumps({"problem_id": "p1", "chosen": "theorem a : True := by trivial"})
        + "\n\n"
        + json.dumps({"problem_id": "p1", "chosen": "theorem b : True := by trivial"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as excinfo:
        bpe._load_jsonl_map(str(path), key="problem_id", value="chosen")

    message = str(excinfo.value)
    assert f"{path}:3" in message
    assert "duplicate problem_id 'p1'" in message
    assert "first seen at line 1" in message


def test_beq_plus_dataset_loader_rejects_duplicate_ids(monkeypatch):
    rows = [
        {"id": "p1", "lean4_formalization": "theorem a : True := by trivial", "lean4_src_header": ""},
        {"id": "p1", "lean4_formalization": "theorem b : True := by trivial", "lean4_src_header": ""},
    ]
    monkeypatch.setattr(bpe, "load_dataset", lambda dataset, split: rows)

    with pytest.raises(ValueError) as excinfo:
        bpe._load_dataset_rows(
            dataset="stub-dataset",
            split="test",
            id_key="id",
            ref_key="lean4_formalization",
            header_key="lean4_src_header",
        )

    message = str(excinfo.value)
    assert "Duplicate 'id' 'p1'" in message
    assert "row 2" in message
    assert "first seen at row 1" in message


def test_beq_plus_resume_stats_reject_duplicate_done_ids(tmp_path):
    path = tmp_path / "beqplus_results.jsonl"
    path.write_text(
        json.dumps({"problem_id": "p1", "a_ok": True})
        + "\n"
        + json.dumps({"problem_id": "p1", "a_ok": False})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as excinfo:
        bpe._load_done_stats(str(path))

    message = str(excinfo.value)
    assert f"{path}:2" in message
    assert "duplicate problem_id 'p1'" in message


def test_beq_plus_eval_select_problem_ids_rejects_mismatched_inputs():
    dataset_rows = {
        "p1": bpe._DatasetRow(ref="theorem p1 : True := by trivial", header=""),
        "dataset_only": bpe._DatasetRow(ref="theorem d : True := by trivial", header=""),
    }
    sel_a = {"p1": "theorem p1 : True := by trivial", "a_only": "theorem a : True := by trivial"}
    sel_b = {"p1": "theorem p1 : True := by trivial", "b_only": "theorem b : True := by trivial"}

    with pytest.raises(ValueError) as excinfo:
        bpe._select_problem_ids(dataset_rows, sel_a, sel_b, a_name="A", b_name="B")

    message = str(excinfo.value)
    assert "problem_id mismatch across selections_a:A, selections_b:B" in message
    assert "a_only" in message
    assert "b_only" in message


def test_beq_plus_eval_select_problem_ids_rejects_selection_ids_missing_dataset():
    dataset_rows = {
        "p1": bpe._DatasetRow(ref="theorem p1 : True := by trivial", header=""),
    }
    sel_a = {"p1": "theorem p1 : True := by trivial", "missing_dataset": "theorem a : True := by trivial"}

    with pytest.raises(ValueError) as excinfo:
        bpe._select_problem_ids(dataset_rows, sel_a)

    message = str(excinfo.value)
    assert "problem_id mismatch across dataset and selections" in message
    assert "missing_dataset" in message


def test_beq_plus_eval_select_problem_ids_can_explicitly_allow_partial_overlap():
    dataset_rows = {
        "p1": bpe._DatasetRow(ref="theorem p1 : True := by trivial", header=""),
        "dataset_only": bpe._DatasetRow(ref="theorem d : True := by trivial", header=""),
    }
    sel_a = {"p1": "theorem p1 : True := by trivial", "a_only": "theorem a : True := by trivial"}
    sel_b = {"p1": "theorem p1 : True := by trivial", "b_only": "theorem b : True := by trivial"}

    assert bpe._select_problem_ids(
        dataset_rows,
        sel_a,
        sel_b,
        allow_partial_overlap=True,
    ) == ["p1"]


def test_beq_plus_oracle_grouped_loader_rejects_duplicate_problem_ids(tmp_path):
    path = tmp_path / "grouped.jsonl"
    path.write_text(
        json.dumps({"problem_id": "p1", "candidates": ["a"]})
        + "\n"
        + json.dumps({"problem_id": "p1", "candidates": ["b"]})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as excinfo:
        bpo._load_grouped_candidates(str(path), "problem_id", "candidates")

    message = str(excinfo.value)
    assert f"{path}:2" in message
    assert "duplicate problem_id 'p1'" in message


def test_beq_plus_oracle_resume_stats_reject_duplicate_done_ids(tmp_path):
    path = tmp_path / "oracle_results.jsonl"
    path.write_text(
        json.dumps({"problem_id": "p1", "oracle_ok": True})
        + "\n"
        + json.dumps({"problem_id": "p1", "oracle_ok": False})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(JsonlError) as excinfo:
        bpo._load_done_stats(str(path))

    message = str(excinfo.value)
    assert f"{path}:2" in message
    assert "duplicate problem_id 'p1'" in message


def test_beq_plus_oracle_select_problem_ids_rejects_mismatched_inputs():
    dataset_rows = {
        "p1": bpe._DatasetRow(ref="theorem p1 : True := by trivial", header=""),
        "pool_only": bpe._DatasetRow(ref="theorem d : True := by trivial", header=""),
    }
    grouped = {"p1": ["theorem p1 : True := by trivial"], "pool_only": ["theorem d : True := by trivial"]}
    sel_a = bpo._SelectionInfo(
        name="A",
        choices={"p1": "theorem p1 : True := by trivial", "a_only": "theorem a : True := by trivial"},
    )

    with pytest.raises(ValueError) as excinfo:
        bpo._select_problem_ids(dataset_rows, grouped, sel_a)

    message = str(excinfo.value)
    assert "problem_id mismatch across candidate_pool, selections_a:A" in message
    assert "pool_only" in message
    assert "a_only" in message


def test_beq_plus_oracle_select_problem_ids_rejects_pool_ids_missing_dataset():
    dataset_rows = {
        "p1": bpe._DatasetRow(ref="theorem p1 : True := by trivial", header=""),
    }
    grouped = {
        "p1": ["theorem p1 : True := by trivial"],
        "missing_dataset": ["theorem missing : True := by trivial"],
    }

    with pytest.raises(ValueError) as excinfo:
        bpo._select_problem_ids(dataset_rows, grouped)

    message = str(excinfo.value)
    assert "problem_id mismatch across dataset and candidate_pool" in message
    assert "missing_dataset" in message


def test_beq_plus_oracle_select_problem_ids_can_explicitly_allow_partial_overlap():
    dataset_rows = {
        "p1": bpe._DatasetRow(ref="theorem p1 : True := by trivial", header=""),
        "dataset_only": bpe._DatasetRow(ref="theorem d : True := by trivial", header=""),
    }
    grouped = {"p1": ["theorem p1 : True := by trivial"], "pool_only": ["theorem d : True := by trivial"]}
    sel_a = bpo._SelectionInfo(
        name="A",
        choices={"p1": "theorem p1 : True := by trivial", "a_only": "theorem a : True := by trivial"},
    )

    assert bpo._select_problem_ids(
        dataset_rows,
        grouped,
        sel_a,
        allow_partial_overlap=True,
    ) == ["p1"]
