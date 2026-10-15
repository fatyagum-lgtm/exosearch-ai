"""ExoSearch command-line interface. Outputs stay local; never assume internet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__
from .data import fetch_toi, load_catalog
from .experiment import infer_saved, train_experiment
from .report import build_dashboard
from .schema import catalog_summary, labeled_and_candidates
from .synthetic import generate_synthetic


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="exosearch", description="TESS TOI catalog-vetting research toolkit (not light-curve discovery)"
    )
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="Download current NASA TOI CSV (internet required)")
    fetch.add_argument("--output", default="data/raw/nasa_toi.csv")
    inspect = sub.add_parser("inspect", help="Validate TOI CSV and summarize statuses")
    inspect.add_argument("--input", required=True)
    train = sub.add_parser("train", help="Train/evaluate Logistic Regression and XGBoost")
    train.add_argument("--input", required=True)
    train.add_argument("--output", default="artifacts/real")
    train.add_argument("--synthetic", action="store_true", help="Mark synthetic input explicitly")
    demo = sub.add_parser("demo", help="Generate labeled synthetic demo and end-to-end report")
    demo.add_argument("--output", default="artifacts/demo")
    demo.add_argument("--stars", type=int, default=350)
    report = sub.add_parser("report", help="Generate offline dashboard from experiment artifacts")
    report.add_argument("--experiment-dir", default="artifacts/real")
    rank = sub.add_parser("rank", help="Show highest-ranked unresolved TOI candidates")
    rank.add_argument("--experiment-dir", default="artifacts/real")
    rank.add_argument("--top", type=int, default=15)
    predict = sub.add_parser("predict", help="Score unresolved candidates with trusted model artifact")
    predict.add_argument("--model", required=True, help="ONLY trusted locally trained joblib model")
    predict.add_argument("--input", required=True)
    predict.add_argument("--output", required=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "fetch":
            result = fetch_toi(args.output)
            print(json.dumps(result, indent=2))
        elif args.command == "inspect":
            df = load_catalog(args.input)
            labeled, unranked = labeled_and_candidates(df)
            print(json.dumps({**catalog_summary(df), "labeled": len(labeled), "unresolved": len(unranked)}, indent=2))
        elif args.command == "train":
            df = load_catalog(args.input)
            result = train_experiment(df, args.input, args.output, is_synthetic=args.synthetic)
            page = build_dashboard(args.output)
            print(f"Model: {result['winner']} | Dashboard: {page}")
        elif args.command == "demo":
            dest = Path(args.output)
            dest.mkdir(parents=True, exist_ok=True)
            csv = dest / "SYNTHETIC_toi_demo.csv"
            generate_synthetic(csv, n_stars=args.stars)
            df = load_catalog(csv)
            result = train_experiment(df, csv, dest, is_synthetic=True)
            page = build_dashboard(dest)
            print(f"SYNTHETIC DEMO ONLY | Model: {result['winner']} | Dashboard: {page}")
            print("These metrics MUST NOT be used as NASA validation or publication evidence.")
        elif args.command == "report":
            print(build_dashboard(args.experiment_dir))
        elif args.command == "rank":
            file = Path(args.experiment_dir) / "candidate_ranking.csv"
            if args.top < 1:
                raise ValueError("--top must be positive")
            df = pd.read_csv(file)
            print(df[["toi", "tid", "tfopwg_disp", "planet_likeness_score"]].head(args.top).to_string(index=False))
        elif args.command == "predict":
            df = load_catalog(args.input)
            unresolved = df.loc[df["tfopwg_disp"].isin(("PC", "APC"))].copy()
            if unresolved.empty:
                print("No PC/APC candidate rows found")
                return 0
            unresolved["planet_likeness_score"] = infer_saved(args.model, unresolved)
            unresolved = unresolved.sort_values("planet_likeness_score", ascending=False)
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            unresolved.to_csv(output, index=False)
            print(f"Saved {len(unresolved)} unconfirmed candidate rankings to {output}")
        return 0
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser().exit(2, f"exosearch: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
