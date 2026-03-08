from __future__ import annotations

import argparse
from pathlib import Path

import tsrl_lite.algorithms  # noqa: F401
import tsrl_lite.encoders  # noqa: F401
import tsrl_lite.envs  # noqa: F401
from tsrl_lite.benchmark import run_benchmark, run_walk_forward_benchmark
from tsrl_lite.builders import build_encoder, build_env, required_future_steps
from tsrl_lite.config import load_experiment_config
from tsrl_lite.data import resolve_price_splits
from tsrl_lite.evaluate import evaluate_agent
from tsrl_lite.export import export_rollouts
from tsrl_lite.matrix import run_benchmark_matrix_spec
from tsrl_lite.optimizer import run_overnight_optimizer, run_overnight_watchdog
from tsrl_lite.pretrain import pretrain_patchtst_backbone
from tsrl_lite.registry import list_agents, list_encoders, list_envs
from tsrl_lite.study import run_study, run_study_spec
from tsrl_lite.trainer import load_trained_agent, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate tsrl-lite experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="run training")
    train_parser.add_argument("--config", required=True, help="path to experiment json config")
    train_parser.add_argument("--output", help="optional output directory override")

    pretrain_parser = subparsers.add_parser(
        "pretrain-patchtst",
        help="run supervised PatchTST backbone pretraining for later RL fine-tuning",
    )
    pretrain_parser.add_argument("--config", required=True, help="path to experiment json config")
    pretrain_parser.add_argument("--output", help="optional output directory override")
    pretrain_parser.add_argument(
        "--task",
        choices=["regime_classification", "future_return_regression"],
        default="regime_classification",
        help="supervised target used during PatchTST pretraining",
    )
    pretrain_parser.add_argument("--epochs", type=int, default=10, help="number of supervised pretraining epochs")
    pretrain_parser.add_argument("--batch-size", type=int, default=64, help="pretraining mini-batch size")
    pretrain_parser.add_argument("--learning-rate", type=float, help="optional pretraining learning rate override")

    benchmark_parser = subparsers.add_parser("benchmark", help="run a multi-seed benchmark")
    benchmark_parser.add_argument("--config", required=True, help="path to experiment json config")
    benchmark_parser.add_argument("--seeds", nargs="+", required=True, type=int, help="list of benchmark seeds")
    benchmark_parser.add_argument("--output", help="optional benchmark root directory")

    walk_forward_parser = subparsers.add_parser("walk-forward", help="run an expanding-window walk-forward benchmark")
    walk_forward_parser.add_argument("--config", required=True, help="path to experiment json config")
    walk_forward_parser.add_argument("--folds", required=True, type=int, help="number of walk-forward folds")
    walk_forward_parser.add_argument(
        "--train-ratio-start",
        type=float,
        default=0.5,
        help="initial fraction of series reserved for the first training fold",
    )
    walk_forward_parser.add_argument("--output", help="optional walk-forward root directory")

    study_parser = subparsers.add_parser("study", help="run and rank multiple experiment configs")
    study_source_group = study_parser.add_mutually_exclusive_group(required=True)
    study_source_group.add_argument("--configs", nargs="+", help="list of experiment config files")
    study_source_group.add_argument("--spec", help="path to a study spec json file")
    study_parser.add_argument(
        "--mode",
        choices=["train", "benchmark", "walk-forward"],
        default="train",
        help="runner used for each config before ranking",
    )
    study_parser.add_argument(
        "--selection-metric",
        default="mean_terminal_score",
        help="metric used to rank the leaderboard",
    )
    study_parser.add_argument(
        "--selection-mode",
        choices=["max", "min"],
        default="max",
        help="whether larger or smaller metric values rank higher",
    )
    study_parser.add_argument(
        "--report-metric",
        action="append",
        default=[],
        help="extra metric to include in the study metrics report, e.g. train.approx_kl",
    )
    study_parser.add_argument("--seeds", nargs="+", type=int, help="seed list for benchmark mode")
    study_parser.add_argument("--folds", type=int, help="fold count for walk-forward mode")
    study_parser.add_argument(
        "--train-ratio-start",
        type=float,
        default=0.5,
        help="initial train ratio for walk-forward study mode",
    )
    study_parser.add_argument(
        "--include-tag",
        action="append",
        default=[],
        help="only run spec variants containing all of these tags",
    )
    study_parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="skip spec variants containing any of these tags",
    )
    study_parser.add_argument("--output", help="optional study root directory")

    matrix_parser = subparsers.add_parser("matrix", help="run a multi-task benchmark matrix from a spec")
    matrix_parser.add_argument("--spec", required=True, help="path to a matrix spec json file")
    matrix_parser.add_argument(
        "--include-tag",
        action="append",
        default=[],
        help="only run task-method variants containing all of these tags",
    )
    matrix_parser.add_argument(
        "--exclude-tag",
        action="append",
        default=[],
        help="skip task-method variants containing any of these tags",
    )
    matrix_parser.add_argument(
        "--no-resume",
        action="store_true",
        help="disable reuse of unchanged per-task study outputs in the target matrix directory",
    )
    matrix_parser.add_argument("--output", help="optional matrix root directory")

    optimizer_parser = subparsers.add_parser(
        "overnight-optimize",
        help="run a resumable overnight config optimizer until a deadline or stop file",
    )
    optimizer_parser.add_argument("--spec", required=True, help="path to optimizer spec json file")
    optimizer_parser.add_argument("--output", help="optional optimizer root directory")
    optimizer_parser.add_argument("--deadline", help="ISO8601 deadline with timezone offset")
    optimizer_parser.add_argument("--max-generations", type=int, help="optional generation limit override")
    optimizer_parser.add_argument("--population-size", type=int, help="optional population size override")
    optimizer_parser.add_argument("--elite-size", type=int, help="optional elite count override")
    optimizer_parser.add_argument("--mutation-rate", type=float, help="optional mutation rate override")
    optimizer_parser.add_argument("--stop-file", help="optional stop file name override")

    watchdog_parser = subparsers.add_parser(
        "overnight-watchdog",
        help="supervise the overnight optimizer and automatically restart after crashes",
    )
    watchdog_parser.add_argument("--spec", required=True, help="path to optimizer spec json file")
    watchdog_parser.add_argument("--output", help="optional optimizer root directory")
    watchdog_parser.add_argument("--deadline", help="ISO8601 deadline with timezone offset")
    watchdog_parser.add_argument("--max-generations", type=int, help="optional generation limit override")
    watchdog_parser.add_argument("--population-size", type=int, help="optional population size override")
    watchdog_parser.add_argument("--elite-size", type=int, help="optional elite count override")
    watchdog_parser.add_argument("--mutation-rate", type=float, help="optional mutation rate override")
    watchdog_parser.add_argument("--stop-file", help="optional stop file name override")
    watchdog_parser.add_argument("--max-restarts", type=int, help="optional watchdog restart limit")
    watchdog_parser.add_argument(
        "--restart-delay-seconds",
        type=float,
        default=5.0,
        help="seconds to wait before relaunching after a crash",
    )

    subparsers.add_parser("list-components", help="list registered envs, encoders, and agents")

    inspect_parser = subparsers.add_parser("inspect-config", help="inspect config-resolved shapes and splits")
    inspect_parser.add_argument("--config", required=True, help="path to experiment json config")

    eval_parser = subparsers.add_parser("evaluate", help="evaluate a saved checkpoint")
    eval_parser.add_argument("--config", required=True, help="path to experiment json config")
    eval_parser.add_argument("--checkpoint", required=True, help="path to .npz checkpoint")
    eval_parser.add_argument("--split", choices=["train", "val", "eval"], default="eval")

    export_parser = subparsers.add_parser("export-rollouts", help="export rollout trajectories to an .npz dataset")
    export_parser.add_argument("--config", required=True, help="path to experiment json config")
    export_parser.add_argument("--checkpoint", required=True, help="path to .npz checkpoint")
    export_parser.add_argument("--output", required=True, help="target .npz file")
    export_parser.add_argument("--split", choices=["train", "val", "eval"], default="eval")
    export_parser.add_argument("--episodes", type=int, default=1)
    export_parser.add_argument("--stochastic", action="store_true", help="sample actions instead of greedy rollout")

    return parser


def handle_train(args: argparse.Namespace) -> None:
    artifacts, summary = train_experiment(args.config, output_dir=args.output)
    print(f"checkpoint: {artifacts.checkpoint_path}")
    if artifacts.best_checkpoint_path is not None:
        print(f"best_checkpoint: {artifacts.best_checkpoint_path}")
    print(f"summary: {artifacts.summary_path}")
    print(f"eval_mean_reward: {summary['evaluation']['mean_reward']:.4f}")


def handle_pretrain_patchtst(args: argparse.Namespace) -> None:
    artifacts, summary = pretrain_patchtst_backbone(
        args.config,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        task_type=args.task,
    )
    print(f"backbone_checkpoint: {artifacts.checkpoint_path}")
    print(f"summary: {artifacts.summary_path}")
    print(f"selection_metric: {summary['selection_metric']}")
    print(f"best_selection_value: {summary['best_selection_value']:.4f}")


def handle_benchmark(args: argparse.Namespace) -> None:
    artifacts, summary = run_benchmark(args.config, seeds=args.seeds, output_dir=args.output)
    print(f"benchmark_summary: {artifacts.summary_path}")
    aggregate = summary["aggregate_evaluation"]
    if "mean_reward" in aggregate:
        print(f"benchmark_mean_reward: {aggregate['mean_reward']['mean']:.4f}")


def handle_walk_forward(args: argparse.Namespace) -> None:
    artifacts, summary = run_walk_forward_benchmark(
        args.config,
        n_folds=args.folds,
        train_ratio_start=args.train_ratio_start,
        output_dir=args.output,
    )
    print(f"walk_forward_summary: {artifacts.summary_path}")
    aggregate = summary["aggregate_evaluation"]
    if "mean_reward" in aggregate:
        print(f"walk_forward_mean_reward: {aggregate['mean_reward']['mean']:.4f}")


def handle_study(args: argparse.Namespace) -> None:
    if args.spec:
        artifacts, summary = run_study_spec(
            args.spec,
            output_dir=args.output,
            include_tags=args.include_tag,
            exclude_tags=args.exclude_tag,
        )
    else:
        artifacts, summary = run_study(
            config_paths=args.configs,
            mode=args.mode,
            output_dir=args.output,
            selection_metric=args.selection_metric,
            selection_mode=args.selection_mode,
            report_metrics=args.report_metric,
            seeds=args.seeds,
            walk_forward_folds=args.folds,
            train_ratio_start=args.train_ratio_start,
        )
    print(f"study_summary: {artifacts.summary_path}")
    print(f"leaderboard_csv: {artifacts.leaderboard_csv_path}")
    print(f"leaderboard_md: {artifacts.leaderboard_md_path}")
    if artifacts.metrics_csv_path is not None:
        print(f"metrics_csv: {artifacts.metrics_csv_path}")
    if artifacts.metrics_md_path is not None:
        print(f"metrics_md: {artifacts.metrics_md_path}")
    if artifacts.pareto_frontier_path is not None:
        print(f"pareto_frontier: {artifacts.pareto_frontier_path}")
    if artifacts.pareto_frontier_md_path is not None:
        print(f"pareto_frontier_md: {artifacts.pareto_frontier_md_path}")
    if artifacts.resolved_config_dir is not None:
        print(f"resolved_configs: {artifacts.resolved_config_dir}")
    print(f"top_experiment: {summary['top_experiment']}")


def handle_matrix(args: argparse.Namespace) -> None:
    artifacts, summary = run_benchmark_matrix_spec(
        args.spec,
        output_dir=args.output,
        include_tags=args.include_tag,
        exclude_tags=args.exclude_tag,
        resume=not args.no_resume,
    )
    print(f"matrix_summary: {artifacts.summary_path}")
    print(f"matrix_leaderboard_csv: {artifacts.leaderboard_csv_path}")
    print(f"matrix_leaderboard_md: {artifacts.leaderboard_md_path}")
    print(f"task_matrix_csv: {artifacts.task_matrix_csv_path}")
    print(f"task_matrix_md: {artifacts.task_matrix_md_path}")
    print(f"pairwise_csv: {artifacts.pairwise_csv_path}")
    print(f"pairwise_md: {artifacts.pairwise_md_path}")
    if artifacts.method_metrics_csv_path is not None:
        print(f"method_metrics_csv: {artifacts.method_metrics_csv_path}")
    if artifacts.method_metrics_md_path is not None:
        print(f"method_metrics_md: {artifacts.method_metrics_md_path}")
    print(f"generated_specs: {artifacts.generated_spec_dir}")
    print(f"resolved_tasks: {artifacts.resolved_task_config_dir}")
    print(f"task_outputs: {artifacts.task_output_dir}")
    print(f"executed_tasks: {summary['executed_task_count']}")
    print(f"resumed_tasks: {summary['resumed_task_count']}")
    if summary.get("matrix_selection_metric") is not None:
        print(f"matrix_selection_metric: {summary['matrix_selection_metric']}")
        print(f"matrix_selection_mode: {summary['matrix_selection_mode']}")
    print(f"top_method: {summary['top_method']}")


def handle_overnight_optimize(args: argparse.Namespace) -> None:
    artifacts, state = run_overnight_optimizer(
        spec_path=args.spec,
        output_dir=args.output,
        deadline=args.deadline,
        max_generations=args.max_generations,
        population_size=args.population_size,
        elite_size=args.elite_size,
        mutation_rate=args.mutation_rate,
        stop_file=args.stop_file,
    )
    print(f"optimizer_state: {artifacts.state_path}")
    print(f"heartbeat: {artifacts.heartbeat_path}")
    print(f"stop_file: {artifacts.stop_path}")
    print(f"status: {state['status']}")
    if state.get("optimizer_kind") is not None:
        print(f"optimizer_kind: {state['optimizer_kind']}")
    if state.get("target_method") is not None:
        print(f"target_method: {state['target_method']}")
    if state.get("best_candidate") is not None:
        print(f"best_experiment: {state['best_candidate']['experiment']}")
        print(f"best_selection_value: {state['best_candidate']['selection_value']:.6f}")


def handle_overnight_watchdog(args: argparse.Namespace) -> None:
    artifacts, state = run_overnight_watchdog(
        spec_path=args.spec,
        output_dir=args.output,
        deadline=args.deadline,
        max_generations=args.max_generations,
        population_size=args.population_size,
        elite_size=args.elite_size,
        mutation_rate=args.mutation_rate,
        stop_file=args.stop_file,
        max_restarts=args.max_restarts,
        restart_delay_seconds=args.restart_delay_seconds,
    )
    print(f"watchdog_state: {artifacts.state_path}")
    print(f"watchdog_heartbeat: {artifacts.heartbeat_path}")
    print(f"optimizer_state: {artifacts.optimizer_state_path}")
    print(f"optimizer_heartbeat: {artifacts.optimizer_heartbeat_path}")
    print(f"stop_file: {artifacts.stop_path}")
    print(f"status: {state['status']}")
    if state.get("optimizer_kind") is not None:
        print(f"optimizer_kind: {state['optimizer_kind']}")
    if state.get("target_method") is not None:
        print(f"target_method: {state['target_method']}")
    if state.get("best_candidate") is not None:
        print(f"best_experiment: {state['best_candidate']['experiment']}")
        print(f"best_selection_value: {state['best_candidate']['selection_value']:.6f}")


def handle_list_components() -> None:
    print("envs:")
    for name in list_envs():
        print(f"  {name}")
    print("encoders:")
    for name in list_encoders():
        print(f"  {name}")
    print("agents:")
    for name in list_agents():
        print(f"  {name}")


def handle_inspect_config(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config)
    splits = resolve_price_splits(
        config=config.data,
        seed=config.seed,
        min_future_steps=required_future_steps(config),
    )
    train_env = build_env(config, prices=splits.train_prices, seed=config.seed, index_offset=splits.train_offset)
    encoder = build_encoder(config, train_env)
    print(f"experiment: {config.experiment_name}")
    print(f"env: {config.env.name}")
    print(f"encoder: {config.encoder.name}")
    print(f"agent: {config.agent.name}")
    print(f"action_dim: {train_env.action_dim}")
    print(f"agent_feature_dim: {train_env.agent_feature_dim}")
    print(f"observation_shape: {encoder.observation_shape}")
    print(f"observation_dim: {encoder.observation_dim}")
    print(f"data_source: {config.data.source}")
    print(f"data_split_mode: {splits.split_mode}")
    if splits.data_start_time is not None:
        print(f"data_start_time: {splits.data_start_time}")
        print(f"data_end_time: {splits.data_end_time}")
    if splits.train_end_time is not None:
        print(f"train_end_time: {splits.train_end_time}")
    if splits.val_end_time is not None:
        print(f"val_end_time: {splits.val_end_time}")
    print(f"train_price_points: {len(splits.train_prices)}")
    print(f"val_price_points: {len(splits.val_prices)}")
    print(f"eval_price_points: {len(splits.eval_prices)}")


def handle_evaluate(args: argparse.Namespace) -> None:
    config = load_experiment_config(args.config)
    splits = resolve_price_splits(
        config=config.data,
        seed=config.seed,
        min_future_steps=required_future_steps(config),
    )
    split_map = {
        "train": (splits.train_prices, splits.train_offset),
        "val": (splits.val_prices, splits.val_offset),
        "eval": (splits.eval_prices, splits.eval_offset),
    }
    eval_prices, eval_offset = split_map[args.split]
    env = build_env(
        config,
        prices=eval_prices,
        seed=config.seed + 1,
        episode_horizon=None,
        random_reset=False,
        index_offset=eval_offset,
    )
    agent = load_trained_agent(
        config=config,
        checkpoint_path=Path(args.checkpoint),
        env=env,
    )
    metrics = evaluate_agent(
        agent=agent,
        env=env,
        episodes=config.trainer.eval_episodes,
        seed=config.seed + 1000,
        random_start=False,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


def handle_export_rollouts(args: argparse.Namespace) -> None:
    artifacts, metadata = export_rollouts(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        split=args.split,
        episodes=args.episodes,
        greedy=not args.stochastic,
    )
    print(f"rollouts: {artifacts.rollout_path}")
    print(f"metadata: {artifacts.metadata_path}")
    print(f"transitions: {metadata['transitions']}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "train":
        handle_train(args)
        return
    if args.command == "pretrain-patchtst":
        handle_pretrain_patchtst(args)
        return
    if args.command == "benchmark":
        handle_benchmark(args)
        return
    if args.command == "walk-forward":
        handle_walk_forward(args)
        return
    if args.command == "study":
        handle_study(args)
        return
    if args.command == "matrix":
        handle_matrix(args)
        return
    if args.command == "overnight-optimize":
        handle_overnight_optimize(args)
        return
    if args.command == "overnight-watchdog":
        handle_overnight_watchdog(args)
        return
    if args.command == "list-components":
        handle_list_components()
        return
    if args.command == "inspect-config":
        handle_inspect_config(args)
        return
    if args.command == "evaluate":
        handle_evaluate(args)
        return
    if args.command == "export-rollouts":
        handle_export_rollouts(args)
        return
    raise ValueError(f"unsupported command: {args.command}")
