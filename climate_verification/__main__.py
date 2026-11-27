import argparse
import json
from pathlib import Path

from .data import load_json
from .paths import RunPaths


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Inspect, train, and evaluate the climate claim verification system."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser(
        "inspect", help="Show one claim from a saved prediction file."
    )
    inspect.add_argument("claim_id", help="Claim ID in the prediction file.")
    inspect.add_argument("--predictions", type=Path, required=True)

    retrieve = commands.add_parser(
        "retrieve", help="Train retrieval models and export evidence IDs."
    )
    classify = commands.add_parser(
        "classify", help="Train and evaluate classifiers on exported evidence."
    )
    run = commands.add_parser(
        "run", help="Run retrieval and classification in sequence."
    )
    for command in (retrieve, classify, run):
        command.add_argument(
            "--data-dir",
            type=Path,
            default=Path("data"),
            help="Directory containing the four input JSON files.",
        )
        command.add_argument(
            "--output-dir",
            type=Path,
            default=Path("outputs/run"),
            help="Directory for generated artifacts.",
        )
    for command in (retrieve, run):
        command.add_argument(
            "--component-ablations",
            action="store_true",
            help="Run retrieval component ablations.",
        )
    retrieve.add_argument(
        "--include-rrf",
        action="store_true",
        help="Export RRF-only evidence for classifier ablations.",
    )
    for command in (classify, run):
        command.add_argument(
            "--all-experiments",
            action="store_true",
            help="Run all four classifier experiments.",
        )
        command.add_argument(
            "--reuse-classifiers",
            action="store_true",
            help="Reuse available checkpoints in the output directory.",
        )

    commands.add_parser("setup-nltk", help="Download the NLTK data used by BM25.")
    evaluate = commands.add_parser(
        "evaluate", help="Score a development prediction file."
    )
    evaluate.add_argument("--predictions", type=Path, required=True)
    evaluate.add_argument("--groundtruth", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "inspect":
        try:
            predictions = load_json(args.predictions)
        except (OSError, ValueError) as error:
            inspect.error(str(error))
        if not isinstance(predictions, dict):
            inspect.error(
                "The prediction file must contain a JSON object keyed by claim ID."
            )
        if args.claim_id not in predictions:
            inspect.error(
                f"Claim {args.claim_id!r} is missing from {args.predictions}."
            )
        print(
            json.dumps(
                {args.claim_id: predictions[args.claim_id]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "setup-nltk":
        from .retrieval.text import download_nltk_data

        download_nltk_data()
        return
    if args.command == "evaluate":
        from .evaluation import evaluate_predictions, print_evaluation

        print_evaluation(evaluate_predictions(args.predictions, args.groundtruth))
        return

    paths = RunPaths(args.data_dir, args.output_dir)
    missing = [
        path for path in (paths.evidence, *paths.claims.values()) if not path.is_file()
    ]
    if missing:
        parser.error("Missing input files: " + ", ".join(map(str, missing)))

    if args.command in {"retrieve", "run"}:
        from .retrieval.pipeline import run_retrieval

        run_retrieval(
            paths,
            component_ablations=args.component_ablations,
            include_rrf=args.include_rrf
            if args.command == "retrieve"
            else args.all_experiments,
        )
    if args.command in {"classify", "run"}:
        from .classification.pipeline import run_classification

        run_classification(
            paths,
            all_experiments=args.all_experiments,
            reuse_classifiers=args.reuse_classifiers,
        )


if __name__ == "__main__":
    main()
