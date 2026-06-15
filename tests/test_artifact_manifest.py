import json
import subprocess
import sys

import pytest

from beqcritic.artifact_manifest import ManifestError, verify_manifest, write_manifest


def test_write_manifest_hashes_nested_run_artifacts(tmp_path):
    run_dir = tmp_path / "quickstart"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "smoke.json").write_text('{"ok": true}\n', encoding="utf-8")
    (run_dir / "logs" / "train.log").write_text("started\n", encoding="utf-8")
    (run_dir / "logs" / "manifest.json").write_text(
        '{"kind": "nested"}\n',
        encoding="utf-8",
    )

    manifest = write_manifest(run_dir)

    manifest_bytes = (run_dir / "manifest.json").read_bytes()
    assert manifest_bytes.endswith(b"\n")
    assert b"\r\n" not in manifest_bytes
    assert manifest["schema_version"] == 2

    paths = {entry["path"] for entry in manifest["artifacts"]}
    assert paths == {"logs/manifest.json", "logs/train.log", "smoke.json"}
    assert (run_dir / "manifest.json").exists()
    verified = verify_manifest(run_dir)
    assert verified["artifact_count"] == 3
    assert verified["run_id"] == "quickstart"


def test_verify_manifest_detects_tampered_artifact(tmp_path):
    run_dir = tmp_path / "quickstart"
    run_dir.mkdir()
    artifact = run_dir / "timing.txt"
    artifact.write_text("train_seconds=1\n", encoding="utf-8")
    write_manifest(run_dir)

    artifact.write_text("train_seconds=2\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="sha256 mismatch"):
        verify_manifest(run_dir)


def test_verify_manifest_rejects_unlisted_extra_file(tmp_path):
    run_dir = tmp_path / "quickstart"
    run_dir.mkdir()
    (run_dir / "ab_metrics.json").write_text("{}\n", encoding="utf-8")
    write_manifest(run_dir)

    (run_dir / "late.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="unexpected artifacts"):
        verify_manifest(run_dir)
    assert verify_manifest(run_dir, allow_extra=True)["artifact_count"] == 1


def test_verify_manifest_rejects_unlisted_nested_manifest(tmp_path):
    run_dir = tmp_path / "quickstart"
    (run_dir / "logs").mkdir(parents=True)
    (run_dir / "ab_metrics.json").write_text("{}\n", encoding="utf-8")
    write_manifest(run_dir)

    (run_dir / "logs" / "manifest.json").write_text(
        '{"kind": "late"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="logs/manifest\\.json"):
        verify_manifest(run_dir)


def test_verify_manifest_rejects_mismatched_run_id(tmp_path):
    run_dir = tmp_path / "quickstart"
    run_dir.mkdir()
    (run_dir / "ab_metrics.json").write_text("{}\n", encoding="utf-8")
    write_manifest(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "other"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="does not match run directory"):
        verify_manifest(run_dir)


def test_verify_manifest_accepts_legacy_v1_without_run_id(tmp_path):
    run_dir = tmp_path / "quickstart"
    run_dir.mkdir()
    (run_dir / "ab_metrics.json").write_text("{}\n", encoding="utf-8")
    manifest = write_manifest(run_dir)
    manifest["schema_version"] = 1
    del manifest["run_id"]
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    verified = verify_manifest(run_dir)

    assert verified["schema_version"] == 1
    assert "run_id" not in verified


def test_verify_manifest_rejects_legacy_v1_mismatched_run_id(tmp_path):
    run_dir = tmp_path / "quickstart"
    run_dir.mkdir()
    (run_dir / "ab_metrics.json").write_text("{}\n", encoding="utf-8")
    manifest = write_manifest(run_dir)
    manifest["schema_version"] = 1
    manifest["run_id"] = "other"
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="does not match run directory"):
        verify_manifest(run_dir)


def test_verify_manifest_rejects_path_escape(tmp_path):
    run_dir = tmp_path / "quickstart"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "quickstart",
                "artifact_count": 1,
                "artifacts": [
                    {
                        "path": "../outside.json",
                        "bytes": 2,
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="run directory"):
        verify_manifest(run_dir)


def test_verify_manifest_rejects_unsupported_schema_version(tmp_path):
    run_dir = tmp_path / "quickstart"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "run_id": "quickstart",
                "artifact_count": 0,
                "artifacts": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match="unsupported manifest schema_version"):
        verify_manifest(run_dir)


def test_manifest_module_cli_writes_and_verifies(tmp_path):
    run_dir = tmp_path / "quickstart"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text("{}\n", encoding="utf-8")

    write = subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.artifact_manifest",
            "--run-dir",
            str(run_dir),
            "--write",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert write.returncode == 0
    write_payload = json.loads(write.stdout)
    assert write_payload["artifact_count"] == 1
    assert write_payload["run_id"] == "quickstart"

    verify = subprocess.run(
        [
            sys.executable,
            "-m",
            "beqcritic.artifact_manifest",
            "--run-dir",
            str(run_dir),
            "--verify",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert verify.returncode == 0
    assert json.loads(verify.stdout)["action"] == "verified"
