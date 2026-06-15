from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any


MANIFEST_NAME = "manifest.json"
SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
SUPPORTED_SCHEMA_VERSIONS = {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}


class ManifestError(ValueError):
    """Raised when a run artifact manifest is invalid or stale."""


def write_manifest(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ManifestError(f"run directory does not exist: {run_dir}")

    artifacts = [
        _artifact_entry(run_dir, path)
        for path in _iter_files(run_dir)
        if not _is_root_manifest(run_dir, path)
    ]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_dir.name,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }
    (run_dir / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return manifest


def verify_manifest(run_dir: str | Path, *, allow_extra: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir)
    manifest_path = run_dir / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise ManifestError(f"missing manifest: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid manifest JSON at {manifest_path}: {exc}") from exc

    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a JSON object")
    schema_version = manifest.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in SUPPORTED_SCHEMA_VERSIONS
    ):
        raise ManifestError(
            f"unsupported manifest schema_version: {schema_version!r}"
        )
    run_id = manifest.get("run_id")
    if schema_version >= SCHEMA_VERSION and (
        not isinstance(run_id, str) or not run_id
    ):
        raise ManifestError("manifest run_id must be a non-empty string")
    if run_id is not None and run_id != run_dir.name:
        raise ManifestError(
            f"manifest run_id {run_id!r} does not match run directory {run_dir.name!r}"
        )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise ManifestError("manifest artifacts must be a list")
    if schema_version >= SCHEMA_VERSION or "artifact_count" in manifest:
        _verify_artifact_count(
            manifest.get("artifact_count"),
            expected=len(artifacts),
        )

    seen_paths: set[str] = set()
    for idx, entry in enumerate(artifacts, 1):
        _verify_entry(run_dir, entry, idx=idx, seen_paths=seen_paths)

    if not allow_extra:
        actual_paths = {
            _relative_path(run_dir, path)
            for path in _iter_files(run_dir)
            if not _is_root_manifest(run_dir, path)
        }
        extra = sorted(actual_paths - seen_paths)
        if extra:
            shown = ", ".join(extra[:5])
            suffix = "" if len(extra) <= 5 else ", ..."
            raise ManifestError(f"unexpected artifacts not in manifest: {shown}{suffix}")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write or verify a BEqCritic run-directory artifact manifest."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--write", action="store_true", help="write manifest.json")
    parser.add_argument("--verify", action="store_true", help="verify manifest.json")
    parser.add_argument(
        "--allow-extra",
        action="store_true",
        help="allow files not listed in manifest.json during verification",
    )
    args = parser.parse_args(argv)

    if bool(args.write) == bool(args.verify):
        parser.error("choose exactly one of --write or --verify")

    try:
        if args.write:
            manifest = write_manifest(args.run_dir)
            action = "wrote"
        else:
            manifest = verify_manifest(args.run_dir, allow_extra=bool(args.allow_extra))
            action = "verified"
    except ManifestError as exc:
        print(f"manifest error: {exc}", file=sys.stderr)
        return 1

    result = {
        "action": action,
        "run_dir": str(Path(args.run_dir)).replace("\\", "/"),
        "artifact_count": int(
            manifest.get("artifact_count", len(manifest["artifacts"]))
        ),
    }
    if manifest.get("run_id"):
        result["run_id"] = str(manifest["run_id"])
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _iter_files(run_dir: Path) -> list[Path]:
    return sorted(path for path in run_dir.rglob("*") if path.is_file())


def _is_root_manifest(run_dir: Path, path: Path) -> bool:
    return path.relative_to(run_dir).as_posix() == MANIFEST_NAME


def _artifact_entry(run_dir: Path, path: Path) -> dict[str, Any]:
    return {
        "path": _relative_path(run_dir, path),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _relative_path(run_dir: Path, path: Path) -> str:
    try:
        rel_path = path.relative_to(run_dir)
    except ValueError as exc:
        raise ManifestError(f"artifact path is outside run directory: {path}") from exc
    return _validate_relative_path(rel_path.as_posix(), where=str(path))


def _validate_relative_path(value: Any, *, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManifestError(f"{where}: artifact path must be a non-empty string")
    if "\\" in value:
        raise ManifestError(f"{where}: artifact path must use forward slashes")
    rel_path = PurePosixPath(value)
    if rel_path.is_absolute():
        raise ManifestError(f"{where}: artifact path must be relative")
    if rel_path.as_posix() != value:
        raise ManifestError(f"{where}: artifact path must be normalized")
    if any(part in {"", ".", ".."} or ":" in part for part in rel_path.parts):
        raise ManifestError(f"{where}: artifact path must stay inside run directory")
    if value == MANIFEST_NAME:
        raise ManifestError(f"{where}: manifest cannot hash itself")
    return value


def _verify_entry(
    run_dir: Path,
    entry: Any,
    *,
    idx: int,
    seen_paths: set[str],
) -> None:
    where = f"manifest artifacts[{idx}]"
    if not isinstance(entry, dict):
        raise ManifestError(f"{where}: entry must be an object")

    rel_path = _validate_relative_path(entry.get("path"), where=where)
    if rel_path in seen_paths:
        raise ManifestError(f"{where}: duplicate artifact path {rel_path!r}")
    seen_paths.add(rel_path)

    path = run_dir.joinpath(*PurePosixPath(rel_path).parts)
    if not path.is_file():
        raise ManifestError(f"{where}: missing artifact {rel_path!r}")

    expected_bytes = entry.get("bytes")
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        raise ManifestError(f"{where}: bytes must be a non-negative integer")
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ManifestError(
            f"{where}: byte size mismatch for {rel_path!r}: "
            f"expected {expected_bytes}, got {actual_bytes}"
        )

    expected_hash = entry.get("sha256")
    if not isinstance(expected_hash, str) or not _looks_like_sha256(expected_hash):
        raise ManifestError(f"{where}: sha256 must be a 64-character hex string")
    actual_hash = _sha256_file(path)
    if actual_hash != expected_hash:
        raise ManifestError(f"{where}: sha256 mismatch for {rel_path!r}")


def _verify_artifact_count(value: Any, *, expected: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ManifestError("manifest artifact_count must be a non-negative integer")
    if value != expected:
        raise ManifestError(
            f"manifest artifact_count {value!r} does not match {expected} artifacts"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _looks_like_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    return all(char in "0123456789abcdef" for char in value.lower())


if __name__ == "__main__":
    raise SystemExit(main())
