from ..data import save_json
from ..evaluation import evaluate_predictions, print_evaluation
from .config import (
    CLASSIFIER_MODELS,
    ENSEMBLE_MODELS,
    ENSEMBLE_WEIGHTS,
    PredictionConfig,
    TrainingConfig,
)
from .inputs import build_classifier_inputs
from .predict import load_candidates, predict
from .train import train

EXPERIMENTS = {
    "top5_ensemble": ("reranker_top5", ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS),
    "max_gap_ensemble": ("reranker_max_gap", ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS),
    "top5_single_distil": ("reranker_top5", ("distil_roberta",), (1.0,)),
    "rrf_top5_ensemble": ("rrf_top5_no_reranker", ENSEMBLE_MODELS, ENSEMBLE_WEIGHTS),
}
EXPERIMENT_DESCRIPTIONS = {
    "top5_ensemble": "Top 5 reranked passages with the weighted ensemble",
    "max_gap_ensemble": "Max-gap evidence selection with the weighted ensemble",
    "top5_single_distil": "Top 5 reranked passages with DistilRoBERTa",
    "rrf_top5_ensemble": "Top 5 RRF passages with the weighted ensemble",
}


def run_classification(paths, *, all_experiments=False, reuse_classifiers=False):
    names = tuple(EXPERIMENTS) if all_experiments else ("max_gap_ensemble",)
    models_by_variant = {}
    for name in names:
        variant, models, _ = EXPERIMENTS[name]
        models_by_variant.setdefault(variant, set()).update(models)

    inputs = build_classifier_inputs(paths, models_by_variant)
    for variant, models in models_by_variant.items():
        for model in sorted(models, key=ENSEMBLE_MODELS.index):
            run_dir = paths.classifier_run(variant, model)
            if reuse_classifiers and load_candidates([run_dir / "summary.json"]):
                print(f"Reusing checkpoint for {variant}/{model}")
                continue

            model_name, epochs, eval_batch_size = CLASSIFIER_MODELS[model]
            print(f"\nTraining {variant}/{model}")
            print("seed | epoch | train loss | elapsed time")
            train(
                TrainingConfig(
                    train=inputs[variant]["train"],
                    dev=inputs[variant]["dev"],
                    run_dir=run_dir,
                    model_name=model_name,
                    epochs=epochs,
                    batch_size=8,
                    eval_batch_size=eval_batch_size,
                    learning_rate=1e-5,
                    max_length=256,
                    seed=29,
                    label_smoothing=0.1,
                )
            )

    rows = []
    for name in names:
        variant, models, weights = EXPERIMENTS[name]
        dev_output = paths.predictions / f"{name}_dev_prediction.json"
        test_output = paths.predictions / f"{name}_test_output.json"
        print(f"\nEvaluating {name}")
        predict(
            PredictionConfig(
                dev=inputs[variant]["dev"],
                test=inputs[variant]["test"],
                dev_output=dev_output,
                test_output=test_output,
                summary_files=[
                    paths.classifier_run(variant, model) / "summary.json"
                    for model in models
                ],
                weights=weights,
                max_length=256,
                batch_size=32,
                max_output_evidence=5,
                cpu=False,
            )
        )
        metrics = evaluate_predictions(dev_output, paths.claims["dev"])
        rows.append(
            {
                "name": name,
                "description": EXPERIMENT_DESCRIPTIONS[name],
                "variant": variant,
                "models": models,
                "weights": weights,
                "dev_prediction": str(dev_output),
                "test_output": str(test_output),
                "metrics": metrics,
                "eval_stdout": print_evaluation(metrics),
            }
        )

    summary_path = paths.predictions / "end_to_end_summary.json"
    save_json(summary_path, rows)
    print(f"Saved end-to-end summary to {summary_path}")
    return rows
