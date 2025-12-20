from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from datasets import Dataset, load_dataset, load_from_disk, concatenate_datasets
import datasets.config


class CachedDatasetNotFound(RuntimeError):
    pass


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dataset_split(dataset: str, split: str) -> Dataset:
    """
    Load a HuggingFace dataset split.

    If the Hub isn't reachable (e.g., offline sandbox) but the dataset has already
    been prepared in the local HF datasets cache, fall back to loading the cached
    Arrow file for the requested split.
    """
    dataset_path = Path(dataset)
    if dataset_path.exists():
        return _load_local_dataset_split(dataset_path, split)
    if not dataset_path.is_absolute():
        from_repo = REPO_ROOT / dataset_path
        if from_repo.exists():
            return _load_local_dataset_split(from_repo, split)

    mapped = _maybe_map_hub_id_to_local(dataset)
    if mapped is not None:
        return _load_local_dataset_split(mapped, split)

    try:
        return load_dataset(dataset, split=split)
    except Exception as original_exc:  # pragma: no cover (varies by environment)
        # Some Hub datasets (including PAug/ProofNetVerif) are published with only
        # valid/test splits. For quickstart-style runs, treat "train" as "valid"
        # when no explicit train split exists, rather than failing.
        if str(split) == "train":
            try:
                return _load_hub_train_fallback(dataset)
            except Exception:
                pass

        try:
            return _load_cached_arrow_split(dataset, split)
        except Exception:
            if str(split) == "train":
                try:
                    return _load_cached_train_fallback(dataset)
                except Exception:
                    pass
            raise original_exc


def _load_local_dataset_split(dataset_path: Path, split: str) -> Dataset:
    # Support datasets saved via `DatasetDict.save_to_disk()`.
    if dataset_path.is_dir() and (dataset_path / "dataset_dict.json").exists():
        dd = load_from_disk(str(dataset_path))
        if split in dd:
            return dd[split]
        if split == "train":
            return _make_train_fallback(dd, dataset_path)
        raise ValueError(
            f"Split {split!r} not available in {dataset_path}; "
            f"available splits: {sorted(dd.keys())}"
        )

    # Support a minimal "parquet files in a folder" layout, e.g.:
    #   hf_datasets/ProofNetVerif/valid-*.parquet, test-*.parquet
    if dataset_path.is_dir():
        parquet_splits = _load_parquet_splits(dataset_path)
        if parquet_splits is not None:
            if split in parquet_splits:
                return parquet_splits[split]
            if split == "train":
                return _make_train_fallback(parquet_splits, dataset_path)
            raise ValueError(
                f"Split {split!r} not available in {dataset_path}; "
                f"available splits: {sorted(parquet_splits.keys())}"
            )

    # Fall back to `load_dataset()` for other local dataset types (data files, scripts, etc).
    return load_dataset(str(dataset_path), split=split)


def _maybe_map_hub_id_to_local(dataset_id: str) -> Optional[Path]:
    """
    Map a Hub dataset id like `owner/name` to a local folder under `hf_datasets/`.

    This is a convenience for offline runs where a dataset has been downloaded or
    exported to disk ahead of time.
    """
    if "/" not in dataset_id:
        return None
    name = dataset_id.split("/", 1)[1]
    for candidate in [
        REPO_ROOT / "hf_datasets" / name,
        REPO_ROOT / "hf_datasets" / dataset_id.replace("/", "--"),
        REPO_ROOT / "offline_bundle" / "hf_datasets" / name,
        REPO_ROOT / "offline_bundle" / "hf_datasets" / dataset_id.replace("/", "--"),
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_parquet_splits(dataset_path: Path) -> Optional[dict[str, Dataset]]:
    valid_files = sorted(dataset_path.glob("valid*.parquet"))
    test_files = sorted(dataset_path.glob("test*.parquet"))
    if not valid_files and not test_files:
        return None

    cache_dir = REPO_ROOT / ".hf_cache" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    splits: dict[str, Dataset] = {}
    if valid_files:
        splits["valid"] = load_dataset(
            "parquet",
            data_files=[str(p) for p in valid_files],
            split="train",
            cache_dir=str(cache_dir),
        )
    if test_files:
        splits["test"] = load_dataset(
            "parquet",
            data_files=[str(p) for p in test_files],
            split="train",
            cache_dir=str(cache_dir),
        )
    return splits


def _make_train_fallback(splits: dict[str, Dataset], dataset_path: Path) -> Dataset:
    if "valid" in splits:
        return splits["valid"]
    if "test" in splits:
        return splits["test"]
    raise ValueError(
        f"Cannot synthesize 'train' split from {dataset_path}; "
        f"available splits: {sorted(splits.keys())}"
    )


def _load_hub_train_fallback(dataset_id: str) -> Dataset:
    """
    Best-effort mapping for Hub datasets without an explicit train split.

    Preference order:
      - valid
      - validation
      - test
    """
    for candidate in ["valid", "validation", "test"]:
        try:
            return load_dataset(dataset_id, split=candidate)
        except Exception:
            continue
    raise ValueError(f"Cannot load a train fallback split for {dataset_id!r}")


def _load_cached_train_fallback(dataset_id: str) -> Dataset:
    """
    Same as `_load_hub_train_fallback`, but using the local HF datasets cache.
    """
    for candidate in ["valid", "validation", "test"]:
        try:
            return _load_cached_arrow_split(dataset_id, candidate)
        except Exception:
            continue
    raise CachedDatasetNotFound(f"No cached valid/validation/test split found for {dataset_id!r}")


def _load_cached_arrow_split(dataset_id: str, split: str) -> Dataset:
    cache_root = Path(datasets.config.HF_DATASETS_CACHE)
    if not cache_root.exists():
        raise CachedDatasetNotFound(f"HF datasets cache not found at {cache_root}")

    candidates = _candidate_cache_dirs(cache_root, dataset_id)
    target_prefix = f"hf://datasets/{dataset_id}@"

    split_error: Optional[Exception] = None
    for ds_dir in candidates:
        for info_path in ds_dir.glob("**/dataset_info.json"):
            try:
                info = json.loads(info_path.read_text())
            except Exception:
                continue

            download_checksums = info.get("download_checksums") or {}
            if not any(
                isinstance(k, str) and k.startswith(target_prefix)
                for k in download_checksums.keys()
            ):
                continue

            available_splits = sorted((info.get("splits") or {}).keys())
            if available_splits and split not in available_splits:
                split_error = ValueError(
                    f"Split {split!r} not available for cached {dataset_id!r}; "
                    f"available splits: {available_splits}"
                )
                continue

            arrow_path = _find_split_arrow_file(info_path.parent, split)
            if not arrow_path:
                split_error = FileNotFoundError(
                    f"Cached dataset found for {dataset_id!r}, but no Arrow file "
                    f"for split {split!r} under {info_path.parent}"
                )
                continue

            return Dataset.from_file(str(arrow_path))

    if split_error:
        raise split_error
    raise CachedDatasetNotFound(f"No cached dataset found for {dataset_id!r}")


def _candidate_cache_dirs(cache_root: Path, dataset_id: str) -> list[Path]:
    if "/" in dataset_id:
        owner = dataset_id.split("/", 1)[0]
        return sorted([p for p in cache_root.glob(f"{owner}___*") if p.is_dir()])
    return sorted([p for p in cache_root.glob(f"{dataset_id}*") if p.is_dir()])


def _find_split_arrow_file(base_dir: Path, split: str) -> Optional[Path]:
    arrow_files = sorted([p for p in base_dir.glob("*.arrow") if p.is_file()])
    for p in arrow_files:
        name = p.name
        if (
            name == f"{split}.arrow"
            or name.endswith(f"-{split}.arrow")
            or name.endswith(f"_{split}.arrow")
            or name.endswith(f".{split}.arrow")
        ):
            return p
    return None
