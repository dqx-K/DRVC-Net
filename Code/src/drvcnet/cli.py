"""Command-line entry points for each reproducibility stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from drvcnet.config import load_config
from drvcnet.data.manifest import build_iemocap_manifest, build_meld_manifest
from drvcnet.data.references import build_reference_index
from drvcnet.data.splits import make_splits
from drvcnet.evaluation.statistics import (
    paired_dialogue_bootstrap,
    paired_permutation_test,
    paired_shift_gain_contrast,
)
from drvcnet.evaluation.interventions import build_reference_intervention
from drvcnet.evaluation.reporting import subgroup_report, summarize_predictions
from drvcnet.io import atomic_write_json, read_jsonl
from drvcnet.preprocessing.detection import extract_face_track_candidates, merge_speaker_assignments
from drvcnet.preprocessing.features import extract_frozen_features
from drvcnet.preprocessing.quality import SharpnessCalibration
from drvcnet.preprocessing.speaker import calibrate_speaker_scores
from drvcnet.preprocessing.perturbations import build_corrupted_observations
from drvcnet.preprocessing.visual import calibrate_sharpness, prepare_visual_observations
from drvcnet.training.engine import evaluate_checkpoint, train_experiment
from drvcnet.training.search import development_search


def build_manifest_main() -> None:
    parser = argparse.ArgumentParser(description="Build a DRVC-Net utterance manifest")
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    iemocap = subparsers.add_parser("iemocap")
    iemocap.add_argument("--root", required=True)
    iemocap.add_argument("--output", required=True)
    meld = subparsers.add_parser("meld")
    meld.add_argument("--root", required=True)
    meld.add_argument("--train-csv", required=True)
    meld.add_argument("--dev-csv", required=True)
    meld.add_argument("--test-csv", required=True)
    meld.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.dataset == "iemocap":
        records = build_iemocap_manifest(args.root, args.output)
    else:
        records = build_meld_manifest(
            args.root, args.train_csv, args.dev_csv, args.test_csv, args.output
        )
    print(json.dumps({"records": len(records), "output": str(Path(args.output).resolve())}))


def make_splits_main() -> None:
    parser = argparse.ArgumentParser(description="Create dialogue-intact dataset splits")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True, choices=["IEMOCAP", "MELD"])
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    paths = make_splits(args.manifest, args.output_dir, args.dataset, args.seed)
    print(json.dumps({"split_files": [str(path) for path in paths]}))


def detect_tracks_main() -> None:
    parser = argparse.ArgumentParser(description="Extract RetinaFace/FAN/ByteTrack candidates")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    records = extract_face_track_candidates(
        args.manifest, args.output_dir, maximum_frames=args.max_frames, device=args.device
    )
    print(json.dumps({"candidates": len(records)}))


def merge_assignments_main() -> None:
    parser = argparse.ArgumentParser(description="Merge frozen target-speaker scores")
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--minimum-score", type=float, default=0.0)
    args = parser.parse_args()
    records = merge_speaker_assignments(
        args.candidates, args.scores, args.output, minimum_score=args.minimum_score
    )
    print(json.dumps({"assignments": len(records)}))


def calibrate_speaker_scores_main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate frozen target-speaker scores")
    parser.add_argument("--audit", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--calibration-output", required=True)
    args = parser.parse_args()
    records = calibrate_speaker_scores(
        args.audit, args.scores, args.output, args.calibration_output
    )
    print(json.dumps({"scores": len(records)}))


def prepare_visual_main() -> None:
    parser = argparse.ArgumentParser(description="Create aligned regional observations")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--minimum-visibility", type=float, default=0.5)
    args = parser.parse_args()
    calibration_value = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibration = SharpnessCalibration(
        float(calibration_value["lower_quantile"]),
        float(calibration_value["upper_quantile"]),
    )
    records = prepare_visual_observations(
        args.manifest,
        args.assignments,
        args.output_dir,
        calibration,
        crop_size=args.crop_size,
        max_frames=args.max_frames,
        minimum_visibility=args.minimum_visibility,
    )
    print(json.dumps({"utterances": len(records)}))


def calibrate_sharpness_main() -> None:
    parser = argparse.ArgumentParser(description="Fit training-only regional sharpness endpoints")
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--split-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--minimum-visibility", type=float, default=0.5)
    parser.add_argument("--lower-percentile", type=float, default=5.0)
    parser.add_argument("--upper-percentile", type=float, default=95.0)
    args = parser.parse_args()
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    calibration = calibrate_sharpness(
        args.assignments,
        {str(value) for value in split["train"]},
        crop_size=args.crop_size,
        minimum_visibility=args.minimum_visibility,
        lower_percentile=args.lower_percentile,
        upper_percentile=args.upper_percentile,
    )
    payload = {
        "lower_quantile": calibration.lower_quantile,
        "upper_quantile": calibration.upper_quantile,
        "lower_percentile": args.lower_percentile,
        "upper_percentile": args.upper_percentile,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def build_perturbation_main() -> None:
    parser = argparse.ArgumentParser(description="Build paired pixel-space stress observations")
    parser.add_argument("--observations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--kind", required=True, choices=["blur", "brow_eye_occlusion", "mouth_occlusion"]
    )
    parser.add_argument("--level", required=True, type=float)
    parser.add_argument("--calibration", required=True)
    args = parser.parse_args()
    value = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibration = SharpnessCalibration(
        float(value["lower_quantile"]), float(value["upper_quantile"])
    )
    records = build_corrupted_observations(
        args.observations,
        args.output_dir,
        kind=args.kind,
        level=args.level,
        calibration=calibration,
    )
    print(json.dumps({"utterances": len(records)}))


def extract_features_main() -> None:
    parser = argparse.ArgumentParser(description="Cache frozen T/A/V features")
    parser.add_argument("--config", required=True)
    parser.add_argument("--visual-observations", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    values = extract_frozen_features(
        load_config(args.config), args.visual_observations, device=args.device
    )
    print(json.dumps({"feature_files": len(values)}))


def build_references_main() -> None:
    parser = argparse.ArgumentParser(description="Build causal dual-reference indices")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    values = build_reference_index(load_config(args.config))
    print(json.dumps({"reference_records": len(values)}))


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train one DRVC-Net fold/seed run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(json.dumps(train_experiment(load_config(args.config), args.device), indent=2))


def search_main() -> None:
    parser = argparse.ArgumentParser(description="Run the equal-budget development search")
    parser.add_argument("--config", required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    print(json.dumps(development_search(load_config(args.config), device_name=args.device), indent=2))


def evaluate_main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one trained checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--device")
    args = parser.parse_args()
    value = evaluate_checkpoint(
        load_config(args.config), args.checkpoint, split=args.split, device_name=args.device
    )
    print(json.dumps(value, indent=2))


def statistics_main() -> None:
    parser = argparse.ArgumentParser(description="Paired dialogue-level comparison")
    parser.add_argument("--first", nargs="+", required=True)
    parser.add_argument("--second", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--metric", default="weighted_f1")
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_926)
    args = parser.parse_args()
    config = load_config(args.config)
    first = [record for path in args.first for record in read_jsonl(path)]
    second = [record for path in args.second for record in read_jsonl(path)]
    bootstrap = paired_dialogue_bootstrap(
        first,
        second,
        config.dataset.labels,
        metric=args.metric,
        repeats=args.repeats,
        seed=args.seed,
        stratify_fold=config.dataset.name == "IEMOCAP",
    )
    permutation = paired_permutation_test(
        first,
        second,
        config.dataset.labels,
        metric=args.metric,
        repeats=args.repeats,
        seed=args.seed,
    )
    payload = {"bootstrap": bootstrap, "permutation": permutation}
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def summarize_main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate the fixed fold/seed protocol")
    parser.add_argument("--predictions", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    records = [record for path in args.predictions for record in read_jsonl(path)]
    payload = {
        "summary": summarize_predictions(
            records,
            config.dataset.labels,
            iemocap_equal_fold_weight=config.dataset.name == "IEMOCAP",
        ),
        "subgroups": subgroup_report(
            records,
            config.dataset.labels,
            ["all", "both", "shift", "stable", "no_previous", "both_shift", "both_stable"],
        ),
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def shift_contrast_main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the Shift-minus-Stable gain contrast")
    parser.add_argument("--first", nargs="+", required=True)
    parser.add_argument("--second", nargs="+", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--population", choices=["all", "both"], default="both")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20_260_926)
    args = parser.parse_args()
    config = load_config(args.config)
    first = [record for path in args.first for record in read_jsonl(path)]
    second = [record for path in args.second for record in read_jsonl(path)]
    payload = paired_shift_gain_contrast(
        first,
        second,
        config.dataset.labels,
        population=args.population,
        repeats=args.repeats,
        seed=args.seed,
        stratify_fold=config.dataset.name == "IEMOCAP",
    )
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


def build_intervention_main() -> None:
    parser = argparse.ArgumentParser(description="Build a frozen reference intervention")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--intervention",
        required=True,
        choices=["N1_other_speaker", "N2_older_random", "N3_self_copy"],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    records = build_reference_intervention(
        load_config(args.config), args.intervention, args.output, seed=args.seed
    )
    eligible = sum(bool(record["intervention_eligible"]) for record in records)
    print(json.dumps({"records": len(records), "eligible": eligible}))


def stress_test_main() -> None:
    from drvcnet.evaluation.stress import stress_test_cli

    stress_test_cli()
