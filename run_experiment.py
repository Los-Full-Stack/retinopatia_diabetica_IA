import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
EXPORT_DIR = PROJECT_ROOT / "export"

MODEL_FILES = [
    "best_retina_model.pth",
    "best_finetuned_model.pth",
    "best_initial_model.pth",
    "last_checkpoint.pth",
]
RESULT_FILES = [
    "metrics.json",
    "clinical_audit.json",
    "inference_policy_audit.json",
    "model_validation.json",
    "classification_report.txt",
]


def load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def copy_if_exists(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def snapshot_current(exp_dir: Path) -> None:
    for name in MODEL_FILES:
        copy_if_exists(MODELS_DIR / name, exp_dir / "before" / "models" / name)
    for name in RESULT_FILES:
        copy_if_exists(RESULTS_DIR / name, exp_dir / "before" / "results" / name)
    if EXPORT_DIR.exists():
        export_snapshot = exp_dir / "before" / "export"
        export_snapshot.mkdir(parents=True, exist_ok=True)
        for file in EXPORT_DIR.iterdir():
            if file.is_file():
                copy_if_exists(file, export_snapshot / file.name)


def snapshot_after(exp_dir: Path) -> None:
    for name in MODEL_FILES:
        copy_if_exists(MODELS_DIR / name, exp_dir / "after" / "models" / name)
    for name in RESULT_FILES:
        copy_if_exists(RESULTS_DIR / name, exp_dir / "after" / "results" / name)
    if EXPORT_DIR.exists():
        export_snapshot = exp_dir / "after" / "export"
        export_snapshot.mkdir(parents=True, exist_ok=True)
        for file in EXPORT_DIR.iterdir():
            if file.is_file():
                copy_if_exists(file, export_snapshot / file.name)


def restore_before(exp_dir: Path) -> None:
    before_models = exp_dir / "before" / "models"
    for name in MODEL_FILES:
        copy_if_exists(before_models / name, MODELS_DIR / name)
    before_results = exp_dir / "before" / "results"
    for name in RESULT_FILES:
        copy_if_exists(before_results / name, RESULTS_DIR / name)
    before_export = exp_dir / "before" / "export"
    if before_export.exists():
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)
        for file in EXPORT_DIR.iterdir():
            if file.is_file() and file.name != ".gitkeep":
                file.unlink()
        for file in before_export.iterdir():
            if file.is_file():
                copy_if_exists(file, EXPORT_DIR / file.name)


def metric_summary(metrics: dict | None, clinical: dict | None) -> dict:
    if not metrics:
        return {}
    recall = metrics.get("recall_per_class", [0, 0, 0, 0, 0])
    summary = {
        "accuracy": metrics.get("accuracy"),
        "balanced_accuracy": metrics.get("balanced_accuracy"),
        "f1_macro": metrics.get("f1_macro"),
        "kappa_quadratic": metrics.get("kappa_quadratic"),
        "recall_grade_2": recall[2] if len(recall) > 2 else None,
        "recall_grade_3": recall[3] if len(recall) > 3 else None,
        "recall_grade_4": recall[4] if len(recall) > 4 else None,
    }
    if clinical:
        clinical_metrics = clinical.get("metrics", {})
        severe = clinical.get("severe_dr_grade_3_or_more", {})
        referable = clinical.get("referable_dr_grade_2_or_more", {})
        summary.update(
            {
                "large_error_rate": clinical_metrics.get("large_error_rate_grade_distance_2_or_more"),
                "referable_sensitivity": referable.get("sensitivity"),
                "severe_sensitivity": severe.get("sensitivity"),
            }
        )
    return summary


def experiment_score(summary: dict) -> float:
    if not summary:
        return -1.0
    difficult_recall = sum(
        float(summary.get(key) or 0.0)
        for key in ("recall_grade_2", "recall_grade_3", "recall_grade_4")
    ) / 3
    large_error = float(summary.get("large_error_rate") or 0.0)
    return (
        0.25 * float(summary.get("f1_macro") or 0.0)
        + 0.20 * float(summary.get("balanced_accuracy") or 0.0)
        + 0.25 * float(summary.get("kappa_quadratic") or 0.0)
        + 0.25 * difficult_recall
        - 0.20 * large_error
    )


def should_keep(before: dict, after: dict) -> tuple[bool, list[str]]:
    reasons = []
    if not after:
        return False, ["No se generaron métricas finales."]

    before_score = experiment_score(before)
    after_score = experiment_score(after)
    if after_score > before_score:
        reasons.append(f"score compuesto subió {before_score:.4f} -> {after_score:.4f}")
    else:
        reasons.append(f"score compuesto no subió {before_score:.4f} -> {after_score:.4f}")

    before_large = before.get("large_error_rate")
    after_large = after.get("large_error_rate")
    large_error_ok = before_large is None or after_large is None or after_large <= before_large
    if not large_error_ok:
        reasons.append(f"errores grandes subieron {before_large:.4f} -> {after_large:.4f}")

    before_kappa = float(before.get("kappa_quadratic") or 0.0)
    after_kappa = float(after.get("kappa_quadratic") or 0.0)
    kappa_ok = after_kappa >= before_kappa - 0.01
    if not kappa_ok:
        reasons.append(f"kappa cayó demasiado {before_kappa:.4f} -> {after_kappa:.4f}")

    return after_score > before_score and large_error_ok and kappa_ok, reasons


def run_step(command: list[str], env: dict, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n\n$ {' '.join(command)}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Falló comando: {' '.join(command)}. Revisa {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta un experimento RetinaAI y conserva solo si mejora.")
    parser.add_argument("--name", default=None, help="Nombre corto del experimento.")
    parser.add_argument("--model-name", default=None, help="efficientnet_b1, efficientnet_b2 o efficientnet_b3.")
    parser.add_argument("--image-size", type=int, default=None, help="Resolución, por ejemplo 300 o 512.")
    parser.add_argument("--initial-epochs", type=int, default=None)
    parser.add_argument("--fine-tune-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sampler-power", type=float, default=None)
    parser.add_argument("--no-weighted-sampler", action="store_true")
    parser.add_argument("--class-multipliers", default=None, help="Ejemplo: 1.0,1.0,1.2,1.6,1.5")
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--no-ordinal-loss", action="store_true")
    parser.add_argument("--ordinal-alpha", type=float, default=None)
    parser.add_argument("--crop-min-scale", type=float, default=None)
    parser.add_argument("--rotation", type=float, default=None)
    parser.add_argument("--color-jitter", type=float, default=None)
    parser.add_argument("--blur-prob", type=float, default=None)
    parser.add_argument("--sharpness-prob", type=float, default=None)
    parser.add_argument("--use-external-clean", action="store_true")
    parser.add_argument("--train-csv", type=Path, default=None, help="CSV de entrenamiento explicito. Activa USE_EXTERNAL_TRAINING_DATA.")
    parser.add_argument("--keep-even-if-worse", action="store_true")
    parser.add_argument("--skip-train", action="store_true", help="Solo evalúa/calibra el checkpoint actual.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = args.name or f"experiment_{stamp}"
    exp_dir = EXPERIMENTS_DIR / f"{stamp}_{name}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    before_metrics = load_json(RESULTS_DIR / "metrics.json")
    before_clinical = load_json(RESULTS_DIR / "clinical_audit.json")
    before_summary = metric_summary(before_metrics, before_clinical)
    snapshot_current(exp_dir)

    env = os.environ.copy()
    options = {
        "RETINAAI_MODEL_NAME": args.model_name,
        "RETINAAI_IMAGE_SIZE": str(args.image_size) if args.image_size else None,
        "RETINAAI_INITIAL_EPOCHS": str(args.initial_epochs) if args.initial_epochs else None,
        "RETINAAI_FINE_TUNE_EPOCHS": str(args.fine_tune_epochs) if args.fine_tune_epochs else None,
        "RETINAAI_BATCH_SIZE": str(args.batch_size) if args.batch_size else None,
        "RETINAAI_RANDOM_SEED": str(args.seed) if args.seed is not None else None,
        "RETINAAI_SAMPLER_WEIGHT_POWER": str(args.sampler_power) if args.sampler_power is not None else None,
        "RETINAAI_USE_WEIGHTED_SAMPLER": "false" if args.no_weighted_sampler else None,
        "RETINAAI_CLASS_WEIGHT_MULTIPLIERS": args.class_multipliers,
        "RETINAAI_FOCAL_LOSS_ENABLED": "true" if args.focal_loss else None,
        "RETINAAI_ORDINAL_LOSS_ENABLED": "false" if args.no_ordinal_loss else None,
        "RETINAAI_ORDINAL_LOSS_ALPHA": str(args.ordinal_alpha) if args.ordinal_alpha is not None else None,
        "RETINAAI_TRAIN_CROP_MIN_SCALE": str(args.crop_min_scale) if args.crop_min_scale is not None else None,
        "RETINAAI_TRAIN_ROTATION_DEGREES": str(args.rotation) if args.rotation is not None else None,
        "RETINAAI_TRAIN_COLOR_JITTER": str(args.color_jitter) if args.color_jitter is not None else None,
        "RETINAAI_TRAIN_GAUSSIAN_BLUR_PROB": str(args.blur_prob) if args.blur_prob is not None else None,
        "RETINAAI_TRAIN_SHARPNESS_PROB": str(args.sharpness_prob) if args.sharpness_prob is not None else None,
    }
    if args.use_external_clean:
        options["RETINAAI_USE_EXTERNAL_TRAINING_DATA"] = "true"
        options["RETINAAI_EXTERNAL_TRAINING_CSV"] = str(
            PROJECT_ROOT / "data" / "splits" / "train_with_external_quality_clean_processed_300_split.csv"
        )
    if args.train_csv:
        train_csv = args.train_csv
        if not train_csv.is_absolute():
            train_csv = PROJECT_ROOT / train_csv
        options["RETINAAI_USE_EXTERNAL_TRAINING_DATA"] = "true"
        options["RETINAAI_EXTERNAL_TRAINING_CSV"] = str(train_csv)

    for key, value in options.items():
        if value is not None:
            env[key] = value

    save_json({"name": name, "env_overrides": {k: v for k, v in options.items() if v is not None}}, exp_dir / "config.json")
    log_path = exp_dir / "run.log"
    commands = []
    if not args.skip_train:
        commands.append([sys.executable, "train.py"])
    commands.extend(
        [
            [sys.executable, "evaluate.py"],
            [sys.executable, "calibrate_model.py"],
            [sys.executable, "evaluate.py"],
            [sys.executable, "clinical_audit.py"],
            [sys.executable, "audit_inference_policy.py"],
        ]
    )

    status = "failed"
    keep = False
    reasons = []
    try:
        for command in commands:
            run_step(command, env, log_path)
        after_metrics = load_json(RESULTS_DIR / "metrics.json")
        after_clinical = load_json(RESULTS_DIR / "clinical_audit.json")
        after_summary = metric_summary(after_metrics, after_clinical)
        keep, reasons = should_keep(before_summary, after_summary)
        if args.keep_even_if_worse:
            keep = True
            reasons.append("keep-even-if-worse activado")
        status = "kept" if keep else "restored"
        snapshot_after(exp_dir)
        if not keep:
            restore_before(exp_dir)
        save_json(
            {
                "status": status,
                "kept": keep,
                "reasons": reasons,
                "before": before_summary,
                "after": after_summary,
                "before_score": experiment_score(before_summary),
                "after_score": experiment_score(after_summary),
            },
            exp_dir / "comparison.json",
        )
    except Exception as exc:
        restore_before(exp_dir)
        save_json(
            {
                "status": status,
                "kept": False,
                "error": str(exc),
                "before": before_summary,
                "before_score": experiment_score(before_summary),
            },
            exp_dir / "comparison.json",
        )
        raise

    comparison = load_json(exp_dir / "comparison.json")
    print(json.dumps(comparison, indent=2, ensure_ascii=False))
    print(f"Experimento guardado en: {exp_dir}")


if __name__ == "__main__":
    main()
