"""Command-line entry points for preparing, reproducing, and demonstrating MusicMap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .data import load_dataset, prepare_data


def main() -> None:
    parser = argparse.ArgumentParser(description="MusicMap: FMA-small music classification lab")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project folder (default: current directory)")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Download/verify metadata and prepare FMA-small")
    prepare.add_argument("--archive", type=Path, help="Existing official fma_metadata.zip")
    prepare.add_argument("--force", action="store_true", help="Rebuild prepared files")
    for name in ("train", "run"):
        command = commands.add_parser(name, help="Run model comparison" if name == "train" else "Prepare, train, analyze, and report")
        command.add_argument("--profile", choices=("quick", "standard"), default="standard")
        command.add_argument("--seed", type=int, default=42)
        command.add_argument("--threads", type=int, default=2)
        if name == "run":
            command.add_argument("--repeats", type=int, default=5, help="Feature permutation repeats")
    improve = commands.add_parser("improve", help="Grouped CV, feature reduction, ANN seed averaging, and ensemble benchmark")
    improve.add_argument("--seed", type=int, default=42)
    improve.add_argument("--threads", type=int, default=2)
    improve.add_argument("--cv-folds", type=int, default=3)
    improve.add_argument("--repeats", type=int, default=5)
    analyze = commands.add_parser("analyze", help="Generate train-only discovery and plots from a completed experiment")
    analyze.add_argument("--threads", type=int, default=2)
    analyze.add_argument("--repeats", type=int, default=5)
    commands.add_parser("report", help="Generate the report from existing experiment and analysis results")
    predict = commands.add_parser("predict", help="Predict using a saved final model")
    source = predict.add_mutually_exclusive_group(required=True)
    source.add_argument("--track-id", type=int)
    source.add_argument("--features-csv", type=Path, help="Flat CSV: first column track_id, followed by the 518 named descriptors")
    predict.add_argument("--output", type=Path, help="Optional prediction CSV")
    neighbors = commands.add_parser("neighbors", help="Retrieve similar training tracks in train-fitted PCA space")
    neighbors.add_argument("--track-id", type=int, required=True)
    neighbors.add_argument("--k", type=int, default=5)
    neighbors.add_argument("--allow-same-artist", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if args.command == "prepare":
            dataset = prepare_data(root, archive=args.archive, force=args.force)
            print(json.dumps(dataset.manifest["split_audit"], indent=2))
        elif args.command == "improve":
            from .improvement import run_improvement
            from .discovery import run_discovery
            from .reporting import generate_report
            dataset = prepare_data(root)
            comparison = run_improvement(
                dataset, root / "reports", seed=args.seed, threads=args.threads, cv_folds=args.cv_folds,
            )
            print(comparison[["model", "cv_mean_f1", "validation_f1_macro", "test_accuracy", "test_f1_macro"]].to_string(index=False))
            run_discovery(dataset, root / "reports", args.seed, args.threads, args.repeats)
            generate_report(dataset, root / "reports")
        elif args.command in ("train", "run"):
            from .experiment import run_experiment
            dataset = prepare_data(root) if args.command == "run" else load_dataset(root)
            comparison = run_experiment(dataset, root / "reports", args.profile, args.seed, args.threads)
            print(comparison[["model", "validation_f1_macro", "test_accuracy", "test_f1_macro"]].to_string(index=False))
            if args.command == "run":
                from .discovery import run_discovery
                from .reporting import generate_report
                run_discovery(dataset, root / "reports", args.seed, args.threads, args.repeats)
                generate_report(dataset, root / "reports")
        elif args.command == "analyze":
            from .discovery import run_discovery
            run_info = json.loads((root / "reports/run_info.json").read_text())
            run_discovery(load_dataset(root), root / "reports", run_info["seed"], args.threads, args.repeats)
        elif args.command == "report":
            from .reporting import generate_report
            generate_report(load_dataset(root), root / "reports")
        elif args.command == "predict":
            _predict(args, root)
        elif args.command == "neighbors":
            from .discovery import nearest_tracks
            dataset = load_dataset(root)
            with threadpool_limits(limits=2):
                result = nearest_tracks(
                    dataset, root / "models", args.track_id, args.k,
                    exclude_same_artist=not args.allow_same_artist,
                )
            print(result.to_string())
    except (FileNotFoundError, ValueError, KeyError) as error:
        parser.exit(2, f"MusicMap: {error}\n")


def _predict(args, root: Path) -> None:
    model = joblib.load(root / "models/best_model.joblib")
    expected = list(model.feature_names_in_)
    if args.track_id is not None:
        dataset = load_dataset(root)
        if args.track_id not in dataset.X.index:
            raise ValueError(f"Track {args.track_id} is not in the prepared subset.")
        X = dataset.X.loc[[args.track_id]]
        result = dataset.metadata.loc[[args.track_id], ["title", "artist", "genre", "split"]].copy()
        result = result.rename(columns={"genre": "actual_genre"})
    else:
        X = pd.read_csv(args.features_csv, index_col=0)
        if set(X.columns) != set(expected):
            missing = sorted(set(expected) - set(X.columns))
            extra = sorted(set(X.columns) - set(expected))
            raise ValueError(f"Feature schema mismatch. Missing: {missing[:5]}; extra: {extra[:5]}.")
        if not X.index.is_unique:
            raise ValueError("Prediction track IDs must be unique.")
        X = X.loc[:, expected].astype(float).replace([np.inf, -np.inf], np.nan)
        if X.empty or X.isna().all(axis=1).any():
            raise ValueError("Prediction data contain no rows or a completely missing feature row.")
        result = pd.DataFrame(index=X.index)
    with threadpool_limits(limits=2):
        result["predicted_genre"] = model.predict(X)
    print(result.to_string())
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output)


if __name__ == "__main__":
    main()
