"""Read the original FMA multi-header CSVs without loading the full feature table."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

FMA_URL = "https://os.unil.cloud.switch.ch/fma/fma_metadata.zip"
FMA_SHA1 = "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"
SPLITS = ("training", "validation", "test")
FEATURE_FAMILIES = {
    "chroma_cens": "Smoothed pitch-class energy; harmonic content",
    "chroma_cqt": "Constant-Q pitch-class energy; harmonic content",
    "chroma_stft": "STFT pitch-class energy; harmonic content",
    "mfcc": "Mel-frequency cepstral coefficients; timbre and spectral envelope",
    "rmse": "Root mean square energy; signal energy",
    "spectral_bandwidth": "Spread of spectral energy around its centroid",
    "spectral_centroid": "Centre of spectral energy; brightness",
    "spectral_contrast": "Peak-to-valley differences in spectral bands",
    "spectral_rolloff": "Frequency below which most spectral energy lies",
    "tonnetz": "Tonal centroid coordinates; harmonic relationships",
    "zcr": "Zero-crossing rate; noisiness and rapid signal changes",
}


def file_hash(path: Path, algorithm: str = "sha256") -> str:
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def download_metadata(destination: Path) -> Path:
    """Download only metadata/features (342 MiB) and verify FMA's published hash."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if file_hash(destination, "sha1") != FMA_SHA1:
            raise ValueError(f"Checksum mismatch for {destination}. Remove it and retry.")
        print("Verified existing FMA metadata archive.", flush=True)
        return destination
    partial = destination.with_suffix(".zip.part")
    for attempt in range(3):
        try:
            request = urllib.request.Request(FMA_URL, headers={"User-Agent": "MusicMap-Lab/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as out:
                digest = hashlib.sha1()
                received = 0
                last_logged = 0
                for block in iter(lambda: response.read(1024 * 1024), b""):
                    out.write(block)
                    digest.update(block)
                    received += len(block)
                    if received - last_logged >= 32 * 1024 * 1024:
                        print(f"Downloaded {received / 1024**2:.0f} MiB...", flush=True)
                        last_logged = received
            if digest.hexdigest() != FMA_SHA1:
                raise ValueError("Downloaded archive does not match FMA's published SHA-1.")
            partial.replace(destination)
            return destination
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("Download did not finish.")


@dataclass
class Dataset:
    X: pd.DataFrame
    metadata: pd.DataFrame
    manifest: dict

    @property
    def y(self) -> pd.Series:
        return self.metadata["genre"]

    def ids(self, split: str) -> pd.Index:
        if split not in SPLITS:
            raise ValueError(f"Unknown split {split!r}; choose from {SPLITS}.")
        return self.metadata.index[self.metadata["split"].eq(split)]


def validate_tables(X: pd.DataFrame, metadata: pd.DataFrame) -> None:
    """Reject ambiguous alignment and accidental metadata-as-feature leakage."""
    required = {"genre", "split", "artist_id", "title", "artist", "album_id"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Missing metadata columns: {sorted(required - set(metadata.columns))}")
    if not X.index.is_unique or not metadata.index.is_unique or not X.columns.is_unique:
        raise ValueError("Track IDs and feature names must be unique.")
    if not X.index.equals(metadata.index):
        raise ValueError("Features and metadata must have identical track IDs in identical order.")
    if X.empty or X.shape[1] == 0:
        raise ValueError("The feature table is empty.")
    if any(not pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes):
        raise ValueError("All predictors must be numeric audio features.")
    if any(name.split("__", 1)[0] not in FEATURE_FAMILIES for name in X.columns):
        raise ValueError("Predictors must belong to the documented FMA audio feature families.")
    if metadata["genre"].isna().any() or metadata["genre"].eq("").any():
        raise ValueError("Every retained track needs a genre.")
    if set(metadata["split"]) != set(SPLITS):
        raise ValueError("All three official splits must be present.")
    if np.isinf(X.to_numpy()).any() or X.isna().all(axis=1).any():
        raise ValueError("Infinite values or completely missing feature rows remain.")
    labels = set(metadata["genre"])
    for split in SPLITS:
        if set(metadata.loc[metadata["split"].eq(split), "genre"]) != labels:
            raise ValueError(f"Split {split} does not contain every genre.")


def split_audit(metadata: pd.DataFrame) -> dict:
    overlaps = {}
    for i, left in enumerate(SPLITS):
        for right in SPLITS[i + 1:]:
            a = set(metadata.loc[metadata["split"].eq(left), "artist_id"].dropna())
            b = set(metadata.loc[metadata["split"].eq(right), "artist_id"].dropna())
            overlaps[f"{left}__{right}"] = len(a & b)
    return {
        "counts": {s: int(metadata["split"].eq(s).sum()) for s in SPLITS},
        "genre_by_split": pd.crosstab(metadata["genre"], metadata["split"]).to_dict(),
        "artist_overlap_counts": overlaps,
        "missing_artist_ids": int(metadata["artist_id"].isna().sum()),
    }


def prepare_data(root: Path, archive: Path | None = None, force: bool = False) -> Dataset:
    root = Path(root)
    processed = root / "data" / "processed"
    if (processed / "manifest.json").exists() and not force:
        print("Using verified prepared FMA-small data.", flush=True)
        return load_dataset(root)
    archive = download_metadata(archive or root / "data/raw/fma_metadata.zip")
    fields = {
        ("track", "genre_top"): "genre",
        ("set", "split"): "split",
        ("artist", "id"): "artist_id",
        ("track", "title"): "title",
        ("artist", "name"): "artist",
        ("album", "id"): "album_id",
    }
    with zipfile.ZipFile(archive) as bundle:
        print("Reading FMA-small track metadata in chunks...", flush=True)
        parts = []
        with bundle.open("fma_metadata/tracks.csv") as stream:
            for chunk in pd.read_csv(
                stream, header=[0, 1], index_col=0, chunksize=5000, low_memory=False
            ):
                selected = chunk.loc[chunk[("set", "subset")].eq("small"), list(fields)].copy()
                selected.columns = list(fields.values())
                parts.append(selected)
        metadata = pd.concat(parts).sort_index()
        if len(metadata) != 8000:
            raise ValueError(f"Expected 8,000 FMA-small tracks, found {len(metadata)}.")
        print("Streaming precomputed features; retaining only the 8,000 selected IDs...", flush=True)
        parts = []
        with bundle.open("fma_metadata/features.csv") as stream:
            for chunk in pd.read_csv(stream, header=[0, 1, 2], index_col=0, chunksize=3000):
                selected = chunk.loc[chunk.index.isin(metadata.index)]
                if not selected.empty:
                    parts.append(selected)
        X = pd.concat(parts).sort_index()
    if not X.index.is_unique or set(X.index) != set(metadata.index):
        raise ValueError("The feature CSV is missing selected IDs or contains duplicate IDs.")
    X = X.loc[metadata.index].astype(np.float32)
    X.columns = ["__".join(map(str, column)) for column in X.columns]
    if X.shape[1] != 518:
        raise ValueError(f"Expected 518 FMA features; found {X.shape[1]}.")
    values = X.to_numpy()
    nonfinite = int((~np.isfinite(values)).sum())
    X = X.replace([np.inf, -np.inf], np.nan)
    unusable = X.index[X.isna().all(axis=1)]
    X = X.drop(index=unusable)
    metadata = metadata.loc[X.index].copy()
    metadata.index.name = X.index.name = "track_id"
    for column in ("artist_id", "album_id"):
        metadata[column] = pd.to_numeric(metadata[column], errors="raise").astype("Int64")
    metadata[["title", "artist"]] = metadata[["title", "artist"]].fillna("")
    validate_tables(X, metadata)
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed / "features.npz",
        values=X.to_numpy(dtype=np.float32),
        track_ids=X.index.to_numpy(dtype=np.int64),
        feature_names=np.array(X.columns, dtype=str),
    )
    metadata.to_csv(processed / "tracks.csv")
    manifest = {
        "dataset": "FMA-small, pre-publication release (2017-05-09)",
        "source_url": FMA_URL,
        "source_sha1": FMA_SHA1,
        "source_checksum_verified": True,
        "license": "CC BY 4.0; Defferrard et al. (2017)",
        "source_small_tracks": 8000,
        "retained_tracks": len(X),
        "feature_count": X.shape[1],
        "storage_dtype": "float32",
        "nonfinite_values_replaced_with_nan": nonfinite,
        "removed_all_missing_track_ids": [int(tid) for tid in unusable],
        "rows_needing_partial_imputation": int(X.isna().any(axis=1).sum()),
        "genres": sorted(metadata["genre"].unique().tolist()),
        "feature_family_sizes": pd.Series(X.columns).str.split("__").str[0].value_counts().to_dict(),
        "split_audit": split_audit(metadata),
        "processed_sha256": {
            name: file_hash(processed / name) for name in ("features.npz", "tracks.csv")
        },
    }
    write_json(processed / "manifest.json", manifest)
    print(f"Prepared {len(X):,} tracks × {X.shape[1]} audio features.", flush=True)
    return Dataset(X, metadata, manifest)


def load_dataset(root: Path) -> Dataset:
    processed = Path(root) / "data/processed"
    manifest_path = processed / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Prepared data not found. Run: musicmap prepare")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["processed_sha256"].items():
        if file_hash(processed / name) != expected:
            raise ValueError(f"Prepared file {name} failed integrity verification. Run prepare --force.")
    with np.load(processed / "features.npz", allow_pickle=False) as arrays:
        X = pd.DataFrame(
            arrays["values"],
            index=pd.Index(arrays["track_ids"], name="track_id"),
            columns=arrays["feature_names"].tolist(),
        )
    metadata = pd.read_csv(
        processed / "tracks.csv", index_col="track_id",
        dtype={"artist_id": "Int64", "album_id": "Int64"}, keep_default_na=False,
    )
    validate_tables(X, metadata)
    return Dataset(X, metadata, manifest)
