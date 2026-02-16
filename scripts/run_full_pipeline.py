#!/usr/bin/env python3
"""
Ad-AIGC-6s Pipeline Orchestrator.

Single entrypoint to regenerate tables, figures, and statistical tests from
released artifacts, and optionally to regenerate intermediate datasets
if you have local access to the underlying videos.

Two common workflows:

1) Fast reproducibility (no videos required)
   - Uses released CSVs under data/metadata/ and predictions under experiments/
   - Regenerates tables/figures and significance tests

2) Full regeneration (videos required; NOT distributed)
   - Recomputes quality metrics and CLIP similarity/labels from videos
   - Rebuilds the filtered main dataset
   - Trains/evaluates the three fusion architectures

Important:
- Copyrighted videos are not distributed with this repository.
- Optional video extraction/generation steps can be run via flags. These steps are compute-intensive and may require downloading model weights (for example, CLIP or diffusion checkpoints), depending on the user environment.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/pipeline_master.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]

def _first_existing(*rel_paths: str) -> Optional[str]:
    """Return the first repo-relative path that exists on disk."""
    for rp in rel_paths:
        if (REPO_ROOT / rp).exists():
            return rp
    return None


def run(cmd: List[str], description: str, env_vars: Optional[Dict[str, str]] = None) -> None:
    """Run a subprocess command while streaming output."""
    logger.info("=" * 60)
    logger.info("STEP: %s", description)
    logger.info("CMD:  %s", " ".join(map(str, cmd)))
    logger.info("=" * 60)

    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        bufsize=1,
        universal_newlines=True,
    )
    assert p.stdout is not None
    for line in p.stdout:
        print(line, end="")
    p.wait()
    if p.returncode != 0:
        raise RuntimeError(f"FAILED (exit={p.returncode}): {description}")
    logger.info("DONE: %s", description)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ad-AIGC-6s pipeline")

    # Optional end-to-end video pipeline (local video extraction/prompts/generation)
    ap.add_argument("--run_download", action="store_true", help="Extract 6s center-window clips from locally available raw videos.")
    ap.add_argument("--max_videos", type=int, default=0, help="Optional cap on number of source ads processed for extraction/prompts/generation (0 = no cap).")
    ap.add_argument("--ensure_annotations", action="store_true", help="Prepare Hussain annotation JSONs into expected layout (used to compute P_real).")
    ap.add_argument("--fix_annotations", action="store_true", help="Alias for --ensure_annotations (kept for backward compatibility).")
    ap.add_argument("--run_prompts", action="store_true", help="Run build_prompts.py to generate data/metadata/ad_prompts.csv.")
    ap.add_argument("--run_generation_pilot", action="store_true", help="Run generate_svd_variants.py in pilot mode (10/20/30/40/50 steps).")
    ap.add_argument("--run_generation_main", action="store_true", help="Run generate_svd_variants.py in main mode (10 steps, variant_id *_optimal).")
    ap.add_argument("--gen_device", type=int, default=0, help="CUDA device id for SVD generation.")

    ap.add_argument("--all", action="store_true", help="Run all steps (requires local videos; not needed for artifact-only reproduction).")
    ap.add_argument("--run_pilot", action="store_true", help="Run pilot ablation + analysis steps.")
    ap.add_argument("--run_main", action="store_true", help="Run main benchmark training/eval + analysis steps.")
    ap.add_argument("--run_robustness", action="store_true", help="Run robustness checks across seeds.")
    ap.add_argument("--run_sensitivity", action="store_true", help="Run lambda sensitivity sweep.")
    ap.add_argument(
        "--run_artifacts",
        action="store_true",
        help=(
            "Regenerate tables and any available figures from released artifacts "
            "(no videos required)."
        ),
    )

    # Heavy recomputation (videos required)
    ap.add_argument("--recompute_quality", action="store_true", help="Recompute OpenCV quality metrics from videos.")
    ap.add_argument("--recompute_similarity", action="store_true", help="Recompute CLIP similarity/labels from videos.")

    # Paths / config
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--annotations_json", default="data/raw/hussain/video/cleaned_result/video_Effective_clean.json")
    ap.add_argument("--main_generation_log", default="data/metadata/main_experiment_generation_log.csv")
    ap.add_argument("--main_quality_csv", default="data/metadata/quality_metrics_main_experiment.csv")
    ap.add_argument("--main_labeled_csv", default="data/metadata/dataset_main_labeled.csv")
    ap.add_argument("--main_dataset_csv", default="data/metadata/ad_aigc_6s_main_experiment.csv")
    ap.add_argument("--main_dataset_cached_csv", default="data/metadata/ad_aigc_6s_main_experiment_cached.csv")

    ap.add_argument("--tables_out_dir", default="paper_artifacts_out")

    args = ap.parse_args()

    # Ensure common directories exist
    Path(REPO_ROOT / "data" / "metadata").mkdir(parents=True, exist_ok=True)
    Path(REPO_ROOT / "logs").mkdir(parents=True, exist_ok=True)

    # Convenience/safety: generating SVD variants requires extracted 6s real clips (or a custom raw-video mode).
    # If generation is requested, ensure extraction runs first (it is idempotent and will skip existing clips).
    if (args.run_generation_pilot or args.run_generation_main) and not args.run_download:
        logger.warning("Generation was requested without --run_download; enabling --run_download automatically.")
        args.run_download = True

    if args.all:
        args.ensure_annotations = True
        args.run_download = True
        args.run_prompts = True
        args.run_generation_pilot = True
        args.run_generation_main = True
        # If we're truly running "all", regenerate metrics/labels from the videos we produced.
        args.recompute_quality = True
        args.recompute_similarity = True

        args.run_pilot = True
        args.run_main = True
        args.run_robustness = True
        args.run_sensitivity = True
        args.run_artifacts = True

    if not any([args.run_pilot, args.run_main, args.run_robustness, args.run_sensitivity, args.run_artifacts]):
        # default to artifact-only (no videos)
        args.run_artifacts = True

    # ---------------------------------------------------------------------
    # Optional: extract/prompts/generation (videos required)
    # ---------------------------------------------------------------------
    if args.fix_annotations:
        args.ensure_annotations = True

    if args.ensure_annotations:
        run([sys.executable, str(REPO_ROOT / "scripts/fix_annotations.py")], "Fix annotations (download/extract JSONs)")

    if args.run_download:
        cmd = [sys.executable, str(REPO_ROOT / "scripts/extract_climax_segments.py")]
        if args.max_videos and args.max_videos > 0:
            cmd.extend(["--max_videos", str(args.max_videos)])
        run(cmd, "Extract 6-second climax segments")

        # If generation is requested, fail fast if no clips were produced/found.
        if args.run_generation_pilot or args.run_generation_main:
            clips = list((REPO_ROOT / "data/raw/hussain_climax").glob("*_real_6s.mp4"))
            if len(clips) == 0:
                raise RuntimeError(
                    "No extracted real clips found under data/raw/hussain_climax after extraction. "
                    "Cannot run SVD generation."
                )

    if args.run_prompts:
        cmd = [sys.executable, str(REPO_ROOT / "scripts/build_prompts.py")]
        if args.max_videos and args.max_videos > 0:
            cmd.extend(["--max_videos", str(args.max_videos)])
        # default benchmark filter: P_real >= 0.5
        cmd.extend(["--min_p_real", "0.5"])
        run(cmd, "Build prompts CSV (filtered by P_real >= 0.5)")

        # If generation is requested, ensure prompts are non-empty.
        if args.run_generation_pilot or args.run_generation_main:
            prompts_path = REPO_ROOT / "data/metadata/ad_prompts.csv"
            if not prompts_path.exists():
                raise FileNotFoundError(f"Prompts CSV was not created: {prompts_path}")
            try:
                import pandas as _pd
                _dfp = _pd.read_csv(prompts_path)
                if len(_dfp) == 0:
                    raise RuntimeError(
                        "ad_prompts.csv has 0 rows after filtering. "
                        "This can happen if P_real >= 0.5 removes all candidates or if no clips were extracted."
                    )
            except Exception:
                # If pandas isn't available for some reason, don't block execution here.
                pass

    if args.run_generation_pilot:
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/generate_svd_variants.py"),
                "--input",
                str(REPO_ROOT / "data/metadata/ad_prompts.csv"),
                "--device",
                str(args.gen_device),
                "--mode",
                "pilot",
            ],
            "Generate SVD variants (pilot: 10/20/30/40/50)",
        )

    if args.run_generation_main:
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/generate_svd_variants.py"),
                "--input",
                str(REPO_ROOT / "data/metadata/ad_prompts.csv"),
                "--device",
                str(args.gen_device),
                "--mode",
                "main",
            ],
            "Generate SVD variants (main: 10 steps, *_optimal)",
        )

    # ---------------------------------------------------------------------
    # PILOT / Ablation
    # ---------------------------------------------------------------------
    if args.run_pilot:
        if args.recompute_quality:
            run(
                [sys.executable, str(REPO_ROOT / "scripts/compute_quality_metrics.py"), "--target", "pilot"],
                "Pilot: compute OpenCV quality metrics (requires pilot videos)",
            )
        run([sys.executable, str(REPO_ROOT / "scripts/ablation_study_analysis.py")], "Pilot: ablation analysis + plot")

    # ---------------------------------------------------------------------
    # MAIN benchmark (optional recomputation + train/eval + baselines/tests)
    # ---------------------------------------------------------------------
    if args.run_main:
        if args.recompute_quality:
            run(
                [sys.executable, str(REPO_ROOT / "scripts/compute_quality_metrics.py"), "--target", "main"],
                "Main: compute OpenCV quality metrics (requires main videos)",
            )

        if args.recompute_similarity:
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/compute_similarity_and_labels.py"),
                    "--quality_csv",
                    str(REPO_ROOT / args.main_quality_csv),
                    "--annotations_json",
                    str(REPO_ROOT / args.annotations_json),
                    "--generation_log_csv",
                    str(REPO_ROOT / args.main_generation_log),
                    "--out_csv",
                    str(REPO_ROOT / args.main_labeled_csv),
                    "--lambda_val",
                    "0.5",
                    "--k_frames",
                    "8",
                ],
                "Main: compute CLIP similarity + labels (requires videos + CLIP weights)",
            )

            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/rebuild_main_dataset_preal.py"),
                    "--input",
                    str(REPO_ROOT / args.main_labeled_csv),
                    "--output",
                    str(REPO_ROOT / args.main_dataset_csv),
                    "--low",
                    "0.50",
                    "--high",
                    "1.00",
                ],
                "Main: rebuild filtered dataset (P_real >= 0.5)",
            )

        # Train/eval single-seed runs (seed 42 by default)
        # Prefer cached CSV (contains precomputed CLIP mid-frame embeddings). If missing,
        # fall back to the non-cached dataset CSV(s).
        train_dataset_csv = _first_existing(args.main_dataset_cached_csv, args.main_dataset_csv, args.main_labeled_csv)
        if train_dataset_csv is None:
            raise FileNotFoundError(
                "No dataset CSV found for training. Expected one of: "
                f"{args.main_dataset_cached_csv}, {args.main_dataset_csv}, {args.main_labeled_csv}"
            )

        for fusion in ["concat", "standard_attn", "quality_gated"]:
            out_dir = REPO_ROOT / "experiments" / f"{fusion}_{args.seed}"
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "main.py"),
                    "--config",
                    str(REPO_ROOT / args.config),
                    "--dataset_csv",
                    str(REPO_ROOT / train_dataset_csv),
                    "--fusion_type",
                    fusion,
                    "--seed",
                    str(args.seed),
                    "--output_dir",
                    str(out_dir),
                    "--num-workers",
                    str(args.num_workers),
                    "--skip_train_if_exists",
                ],
                f"Train/eval: {fusion} (seed={args.seed})",
            )

        # Baselines, significance, gating plot
        baselines_dataset_csv = _first_existing(args.main_dataset_csv, args.main_labeled_csv, args.main_dataset_cached_csv)
        if baselines_dataset_csv is None:
            raise FileNotFoundError(
                "No dataset CSV found for baselines. Expected one of: "
                f"{args.main_dataset_csv}, {args.main_labeled_csv}, {args.main_dataset_cached_csv}"
            )
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/build_missing_baselines.py"),
                "--dataset_csv",
                str(REPO_ROOT / baselines_dataset_csv),
                "--out_root",
                str(REPO_ROOT / "experiments/baselines"),
                "--seed",
                str(args.seed),
            ],
            "Baselines: build deterministic baselines (test set only)",
        )

        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/calc_significance.py"),
                "--proposed",
                str(REPO_ROOT / "experiments" / f"quality_gated_{args.seed}" / "test_predictions.csv"),
                "--baseline",
                str(REPO_ROOT / "experiments" / f"concat_{args.seed}" / "test_predictions.csv"),
                "--n_boot",
                "10000",
                "--seed",
                str(args.seed),
            ],
            "Significance: paired bootstrap (quality_gated vs concat)",
        )

        run([sys.executable, str(REPO_ROOT / "scripts/generate_gating_plot.py")], "Figure 7: generate gating plot")

    # ---------------------------------------------------------------------
    # Robustness
    # ---------------------------------------------------------------------
    if args.run_robustness:
        robustness_dataset_csv = _first_existing(args.main_dataset_cached_csv, args.main_dataset_csv, args.main_labeled_csv)
        if robustness_dataset_csv is None:
            raise FileNotFoundError(
                "No dataset CSV found for robustness. Expected one of: "
                f"{args.main_dataset_cached_csv}, {args.main_dataset_csv}, {args.main_labeled_csv}"
            )
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/run_robustness_check.py"),
                "--config",
                args.config,
                "--dataset_csv",
                robustness_dataset_csv,
                "--num_workers",
                str(args.num_workers),
            ],
            "Robustness: 3 fusions × 3 seeds",
        )

    # ---------------------------------------------------------------------
    # Lambda sensitivity (requires data/metadata/lambda_sensitivity/*.csv)
    # ---------------------------------------------------------------------
    if args.run_sensitivity:
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/run_lambda_sensitivity.py"),
                "--config",
                args.config,
                "--datasets_dir",
                "data/metadata/lambda_sensitivity",
                "--out_root",
                "experiments/lambda_sensitivity",
                "--num_workers",
                str(args.num_workers),
            ],
            "Sensitivity: lambda sweep",
        )

    # ---------------------------------------------------------------------
    # Tables/Figures
    # ---------------------------------------------------------------------
    if args.run_artifacts:
        # Ensure figure output directory exists (even if some figures are skipped)
        figures_out_dir = REPO_ROOT / "outputs" / "paper_artifacts"
        figures_out_dir.mkdir(parents=True, exist_ok=True)

        tables_dataset_csv = _first_existing(args.main_dataset_cached_csv, args.main_dataset_csv, args.main_labeled_csv)
        if tables_dataset_csv is None:
            logger.error(
                "Skipping table/figure generation: no dataset CSV found (expected one of %s, %s, %s)",
                args.main_dataset_cached_csv, args.main_dataset_csv, args.main_labeled_csv,
            )
            return

        pred_csvs = [
            str(REPO_ROOT / "experiments" / f"concat_{args.seed}" / "test_predictions.csv"),
            str(REPO_ROOT / "experiments" / f"standard_attn_{args.seed}" / "test_predictions.csv"),
            str(REPO_ROOT / "experiments" / f"quality_gated_{args.seed}" / "test_predictions.csv"),
            str(REPO_ROOT / "experiments" / "baselines" / "mean_pred" / "test_predictions.csv"),
            str(REPO_ROOT / "experiments" / "baselines" / "q_only_lr" / "test_predictions.csv"),
            str(REPO_ROOT / "experiments" / "baselines" / "s_only_lr" / "test_predictions.csv"),
            str(REPO_ROOT / "experiments" / "baselines" / "q_s_lr" / "test_predictions.csv"),
        ]
        run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/generate_tables_and_figures.py"),
                "--pred_csvs",
                *pred_csvs,
                "--dataset_csv",
                str(REPO_ROOT / tables_dataset_csv),
                "--out_dir",
                str(REPO_ROOT / args.tables_out_dir),
                "--abs_tol",
                "0.1",
                "--include_oracle",
                "--oracle_features",
                "p_q_s",
                "--correlation_target",
                "retention_factor",
                "--correlation_metrics",
                "vbench_motion_smoothness",
                "vbench_dynamic_degree",
                "vbench_temporal_flicker",
                "vbench_sharpness_consistency",
            ],
            "Tables and correlation table",
        )

        # Figures: gating plot (requires a trained checkpoint)
        gating_ckpt = REPO_ROOT / "experiments" / f"quality_gated_{args.seed}" / "best_model.pt"
        if gating_ckpt.exists():
            run(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/generate_gating_plot.py"),
                    "--config",
                    str(REPO_ROOT / args.config),
                    "--model_path",
                    str(gating_ckpt),
                    "--out_dir",
                    str(figures_out_dir),
                    "--fig_name",
                    "figure_7_final.png",
                    "--split_seed",
                    "42",
                ],
                "Figure 7: generate gating plot",
            )
        else:
            logger.warning("Skipping gating plot (checkpoint not found): %s", gating_ckpt)

        # Figures: pilot ablation elbow plot (requires pilot metrics CSV)
        pilot_metrics = REPO_ROOT / "data" / "metadata" / "quality_metrics_pilot.csv"
        if pilot_metrics.exists():
            run([sys.executable, str(REPO_ROOT / "scripts/ablation_study_analysis.py")], "Pilot: ablation analysis + plot")
        else:
            logger.info("Skipping pilot ablation plot (missing): %s", pilot_metrics)

    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("Tables: %s/", args.tables_out_dir)
    logger.info("Figures: outputs/paper_artifacts/")
    logger.info("Inputs used: experiments/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()