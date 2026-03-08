# tsrl-lite

`tsrl-lite` is a lightweight, plug-and-play time-series reinforcement learning framework built for research and reuse.
It is intentionally decoupled from LLM-specific RL stacks so it can be applied directly to time-series tasks and time-series foundation models.
The project focuses on three things:

- a small but extensible project structure
- pluggable time-series state encoders
- runnable baselines from config to checkpoint

## What is included

- Synthetic or CSV-based price series loading
- Sliding-window time-series environments
- Pluggable encoders for turning raw temporal windows into RL observations
- Multi-asset CSV and synthetic series support for portfolio-style RL tasks
- A precomputed embedding adapter for time-series foundation-model outputs
- Linear actor-critic and linear PPO baselines
- An optional trainable GRU + PPO sequence policy path
- An optional trainable DLinear + PPO sequence policy path
- An optional trainable PatchTST-style + PPO sequence policy path
- Training and evaluation CLI
- Multi-seed and walk-forward benchmark workflows
- Multi-config study runner with leaderboard exports
- Multi-task benchmark matrix with per-task studies and aggregate method rankings
- Open-source friendly project files and smoke tests
- Smoke tests for the end-to-end loop
- Time-aware CSV loading and explicit timestamp-boundary splits
- Trading-aware risk metrics such as Sharpe ratio and max drawdown

## Project layout

```text
tsrl-lite/
  configs/
  src/tsrl_lite/
    algorithms/
    encoders/
    data/
    envs/
    networks/
  tests/
```

## Quick start

Create an isolated environment and install the package:

```bash
cd /home/nixiak/tsrl-lite
uv venv
uv pip install -e .
```

For the optional trainable sequence-model path:

```bash
uv pip install -e ".[torch]"
```

Train on the bundled synthetic config:

```bash
uv run tsrl-train train --config configs/synthetic_trading.json
```

Train on the built-in regime classification task:

```bash
uv run tsrl-train train --config configs/synthetic_regime.json
```

Train on the built-in multi-asset portfolio task:

```bash
uv run tsrl-train train --config configs/synthetic_portfolio.json
```

Train with the optional GRU + PPO sequence model:

```bash
uv run tsrl-train train --config configs/synthetic_regime_gru.json
```

Train with the optional DLinear + PPO sequence model:

```bash
uv run tsrl-train train --config configs/synthetic_regime_dlinear.json
```

Train with the optional PatchTST-style + PPO sequence model:

```bash
uv run tsrl-train train --config configs/synthetic_regime_patchtst.json
```

Train PatchTST on the built-in multi-asset portfolio task with channel-independent multivariate tokenization:

```bash
uv run tsrl-train train --config configs/synthetic_portfolio_patchtst.json
```

Train with the auxiliary masked-patch version of PatchTST + PPO:

```bash
uv run tsrl-train train --config configs/synthetic_regime_patchtst_aux.json
```

Pretrain a PatchTST backbone with supervised regime labels before RL fine-tuning:

```bash
uv run tsrl-train pretrain-patchtst \
  --config configs/synthetic_regime_patchtst_aux.json \
  --output runs/synthetic_regime_patchtst_pretrain
```

Or pretrain it on continuous future-return regression targets:

```bash
uv run tsrl-train pretrain-patchtst \
  --config configs/synthetic_regime_patchtst_aux.json \
  --task future_return_regression \
  --output runs/synthetic_regime_patchtst_return_pretrain
```

Or jointly pretrain the same backbone on both regime labels and future-return regression:

```bash
uv run tsrl-train pretrain-patchtst \
  --config configs/synthetic_regime_patchtst_aux.json \
  --task joint_regime_return \
  --output runs/synthetic_regime_patchtst_joint_pretrain
```

Then fine-tune from the saved backbone checkpoint:

```bash
uv run tsrl-train train --config configs/synthetic_regime_patchtst_finetune_template.json
```

The fine-tune template now uses a short backbone warmup: it freezes the pretrained PatchTST backbone for the first few PPO updates and then automatically unfreezes it through `agent.params.unfreeze_backbone_after_updates`.

Train with the optional Transformer + PPO sequence model:

```bash
uv run tsrl-train train --config configs/synthetic_regime_transformer.json
```

Train on a real CSV series with explicit train/validation time boundaries:

```bash
uv run tsrl-train train --config configs/csv_time_trading_template.json
```

Train on a multi-asset CSV with explicit time boundaries:

```bash
uv run tsrl-train train --config configs/csv_multiasset_portfolio_template.json
```

Generate demo embeddings and plug them in as an encoder:

```bash
uv run python scripts/generate_demo_embeddings.py \
  --config configs/synthetic_trading.json \
  --output data/demo_embeddings.npy \
  --dim 16
```

Evaluate the saved checkpoint:

```bash
uv run tsrl-train evaluate \
  --config configs/synthetic_trading.json \
  --checkpoint runs/synthetic_trading/agent_checkpoint.npz
```

Evaluate on the validation split or use the best checkpoint selected during training:

```bash
uv run tsrl-train evaluate \
  --config configs/synthetic_trading.json \
  --checkpoint runs/trading_valbest_cli/best_checkpoint.npz \
  --split val
```

Run a multi-seed benchmark:

```bash
uv run tsrl-train benchmark \
  --config configs/synthetic_trading.json \
  --seeds 7 11 19
```

Run an expanding-window walk-forward benchmark:

```bash
uv run tsrl-train walk-forward \
  --config configs/synthetic_trading.json \
  --folds 3 \
  --train-ratio-start 0.5
```

Run a multi-config study and export a leaderboard:

```bash
uv run tsrl-train study \
  --configs configs/synthetic_trading.json configs/synthetic_regime.json \
  --mode train \
  --selection-metric mean_terminal_score
```

Include training-stability diagnostics in that study report:

```bash
uv run tsrl-train study \
  --configs configs/synthetic_trading.json configs/synthetic_regime.json \
  --mode train \
  --selection-metric mean_terminal_score \
  --report-metric train.approx_kl \
  --report-metric train.value_loss \
  --report-metric train_tail.clip_fraction
```

Run a spec-driven experiment matrix from a single JSON file:

```bash
uv run tsrl-train study \
  --spec configs/study_trading_encoder_matrix.json
```

Run a risk-aware study that maximizes Sharpe ratio while enforcing a max drawdown cap:

```bash
uv run tsrl-train study \
  --spec configs/study_trading_risk_aware.json
```

That study also writes:

- `metrics_report.csv` / `metrics_report.md`: compact multi-metric comparison table
- `pareto_frontier.json` / `pareto_frontier.md`: non-dominated experiments under the configured Pareto metrics

Run a multi-task benchmark matrix that compares the same method family across different tasks and aggregates an overall method leaderboard:

```bash
uv run tsrl-train matrix \
  --spec configs/matrix_synthetic_research_suite.json
```

Run a weighted cross-task sequence benchmark over trading, regime, and portfolio tasks:

```bash
uv run tsrl-train matrix \
  --spec configs/matrix_cross_task_sequence_suite.json
```

Run a stability-aware matrix that reranks methods by aggregated `train.approx_kl` while enforcing a matrix-level feasibility cap:

```bash
uv run tsrl-train matrix \
  --spec configs/matrix_cross_task_stability_suite.json
```

Rerunning the same matrix command in the same output directory now reuses unchanged per-task study outputs automatically. Use `--no-resume` if you want to force a full rerun.

That matrix writes:

- `matrix_leaderboard.csv` / `matrix_leaderboard.md`: aggregate method ranking across tasks
- `task_matrix.csv` / `task_matrix.md`: per-method breakdown with one slice per task
- `pairwise_wins.csv` / `pairwise_wins.md`: weighted head-to-head method comparisons across all tasks
- `method_metrics.csv` / `method_metrics.md`: weighted aggregation of shared task `report_metrics`, including shared `train.*` stability metrics when present
- `generated_specs/`: generated per-task study specs
- `tasks/*`: full per-task study outputs, including their own leaderboards and Pareto reports

Matrix specs also support `task_weight`, so high-priority tasks can contribute more to the aggregate leaderboard than auxiliary tasks.
They now also support `matrix_selection_metric`, `matrix_selection_mode`, and `matrix_metric_constraints` for method-level reranking on top of the aggregated `method_metrics`.

Run only a tagged subset of a larger study spec:

```bash
uv run tsrl-train study \
  --spec configs/study_trading_encoder_matrix.json \
  --include-tag trading \
  --include-tag encoder
```

Run the resumable overnight optimizer until a deadline:

```bash
uv run tsrl-train overnight-optimize \
  --spec configs/overnight_trading_search.json \
  --deadline 2026-03-08T10:00:00+08:00 \
  --output runs/overnight_trading_search
```

Run a risk-aware overnight search that only prefers candidates satisfying the drawdown constraint in the spec:

```bash
uv run tsrl-train overnight-optimize \
  --spec configs/overnight_trading_risk_aware_search.json \
  --deadline 2026-03-08T10:00:00+08:00 \
  --output runs/overnight_trading_risk_aware
```

Tune one method inside a cross-task matrix and rank candidates by the aggregated method-level metric:

```bash
uv run tsrl-train overnight-optimize \
  --spec configs/overnight_matrix_gru_search.json \
  --max-generations 1 \
  --output runs/overnight_matrix_gru_search
```

Run the supervised watchdog wrapper if you want automatic relaunch after Python-level crashes:

```bash
uv run tsrl-train overnight-watchdog \
  --spec configs/overnight_trading_search.json \
  --deadline 2026-03-08T10:00:00+08:00 \
  --output runs/overnight_trading_search
```

Or launch the watchdog in the background with the helper script:

```bash
bash scripts/run_overnight.sh \
  configs/overnight_trading_search.json \
  runs/overnight_trading_search \
  2026-03-08T10:00:00+08:00
```

During long unattended runs, the optimizer continuously updates:

- `state.json`: full optimizer state, including the active generation plan
- `heartbeat.json`: lightweight liveness snapshot for dashboards or polling
- `progress.csv`: one row per evaluated candidate across all generations
- `best_config.json`: the current best resolved experiment config, or the resolved matrix spec in matrix-tuning mode
- `best_candidate.json`: the current best candidate metadata

If the process crashes mid-generation, rerunning the same command resumes the unfinished candidate plan instead of discarding the partial generation.

For real datasets, the CSV loader supports:

- `data.timestamp_column`: parse and sort rows chronologically
- `data.timestamp_format`: optional `strptime` format if the file is not ISO-8601
- `data.price_columns`: load multiple aligned asset columns into a single time-series tensor
- `data.start_time` / `data.end_time`: crop the usable date range before training
- `data.train_end_time` / `data.val_end_time`: use explicit chronological split boundaries instead of ratio splits

This makes the same training and benchmarking pipeline usable for both synthetic studies and fixed historical backtests.

Export rollout trajectories for offline analysis or downstream training:

```bash
uv run tsrl-train export-rollouts \
  --config configs/synthetic_trading.json \
  --checkpoint runs/ppo_trading/agent_checkpoint.npz \
  --output runs/ppo_trading/eval_rollouts.npz \
  --split eval \
  --episodes 1
```

Trading evaluations now expose risk-aware metrics such as:

- `mean_sharpe_ratio`
- `mean_sortino_ratio`
- `mean_max_drawdown`
- `mean_terminal_return`
- `mean_calmar_ratio`

These can be used directly as `trainer.selection_metric` in configs, for example pairing `mean_max_drawdown` with `selection_mode = "min"` or `mean_sharpe_ratio` with `selection_mode = "max"`.

Study and overnight optimizer specs also support `metric_constraints`, for example:

```json
{
  "selection_metric": "mean_sharpe_ratio",
  "selection_mode": "max",
  "metric_constraints": {
    "mean_max_drawdown": {"max": 0.35}
  }
}
```

Constraint-aware ranking always prefers feasible runs first. If no run is feasible yet, the least-violating candidate is ranked highest.

When `pareto_metrics` are present in a study or optimizer spec, study outputs also compute a Pareto frontier over the feasible runs if any exist, otherwise over the full set.

Exported rollout datasets now also include step-level numeric info traces such as `info_equity`, `info_turnover`, `info_market_return`, and matching `metadata.info_keys`.

The optional `torch-gru-ppo`, `torch-dlinear-ppo`, `torch-patchtst-ppo`, and `torch-transformer-ppo` agents now support PPO minibatches, `target_kl` early stopping, and clipped value updates through `agent.params`. Their training summaries also expose `train_update_metrics` so you can inspect signals such as `approx_kl`, `clip_fraction`, `explained_variance`, and `early_stop_triggered` without opening the full history file. The PatchTST agent also supports masked-patch auxiliary learning through `aux_loss_coef`, `aux_mask_ratio`, and `aux_epochs`, and now exposes a `channel_independent` mode for multivariate windows so portfolio-style time series can be tokenized closer to the original PatchTST design.

There is now also a lightweight supervised pretraining path for PatchTST. `tsrl-train pretrain-patchtst` can train the PatchTST backbone on regime classification labels, future-return regression targets, or a joint multitask objective derived from the same time-series windows. It saves `backbone_checkpoint.pt`, and the RL agent can then load it through `agent.params.pretrained_backbone_path`. Fine-tuning can either keep the backbone trainable from the start, freeze it with `freeze_backbone`, or do a more realistic warm start by freezing first and then releasing it later with `unfreeze_backbone_after_updates`. If you need to rebalance the multitask objective, `agent.params.pretrain_classification_loss_coef` and `agent.params.pretrain_regression_loss_coef` control the two supervised heads during pretraining.

Those training-stability signals are now also available inside study and matrix specs. For example, `report_metrics` or `selection_metric` can reference names such as `train.approx_kl`, `train.value_loss`, `train_tail.clip_fraction`, or `validation.mean_reward`.

The trainer now supports periodic validation, best-checkpoint selection, `history.csv`, optional early stopping through `trainer.*` config fields, reusable benchmark/study paths built on the same training loop, study-spec resolution into reproducible generated configs, spec-driven grids with tag filtering, multi-task benchmark matrices with per-task aggregation, and a resumable overnight optimizer with deadline, stop-file, progress snapshots, crash recovery, and watchdog supervision.

Run smoke tests:

```bash
uv run python -m unittest discover -s tests -v
```

## Design notes

- `data`: data sources and train/eval splitting
- `envs`: reusable time-series decision environments
- `encoders`: state adapters for raw windows, handcrafted features, multi-asset context features, or future foundation-model embeddings
- `networks`: policy/value function implementations
- `algorithms`: learning rules and agent wrappers
- `builders.py`: component construction from config and registries
- `trainer.py`: rollout, training loop, checkpointing
- `benchmark.py`: multi-seed and walk-forward reproducibility with aggregate reporting
- `study.py`: cross-config experiment comparison with CSV and Markdown leaderboards
- `matrix.py`: multi-task benchmark matrix orchestration with task-wise studies and aggregate method rankings
- `metrics_report.*`: compact per-study metric matrix for key metrics
- `pareto_frontier.*`: non-dominated experiment slice for multi-objective analysis
- `optimizer.py`: deadline-driven overnight config search with resume, progress snapshots, stop-file support, and watchdog supervision
- `configs/study_*.json`: sample spec files for experiment matrices
- `configs/matrix_*.json`: sample multi-task benchmark matrix specs
- `configs/overnight_*.json`: sample optimizer specs
- `metric_constraints`: optional feasibility filters for study and optimizer specs
- `configs/csv_time_*.json`: template configs for real CSV datasets with timestamp boundaries
- `export.py`: rollout dataset export for downstream model work

## Why this matches the project goal

- Lightweight: only `numpy` is required for the baseline framework.
- Plug-and-play: `env`, `encoder`, and `agent` are all selected by config and registry.
- Time-series first: the framework operates on raw temporal windows and task-specific rewards instead of LLM-oriented token pipelines.
- Time-series-backbone ready: besides GRU and Transformer, it now includes a lightweight DLinear policy path that matches the inductive bias of classic temporal decomposition models.
- Foundation-model aligned: it now also includes a PatchTST-style policy path, including channel-independent multivariate tokenization, so patch-based temporal modeling can be studied without coupling the framework to a specific external model stack.
- Representation-learning ready: the PatchTST path can now mix PPO with masked patch reconstruction, which starts to bridge RL optimization and time-series self-supervised learning inside the same framework.
- Pretrain-and-finetune ready: PatchTST backbones can now be pretrained on supervised temporal labels, loaded back into RL experiments, and fine-tuned with a frozen-head warmup followed by automatic backbone unfreezing.
- Multitask pretraining ready: the same PatchTST pretraining entrypoint now supports discrete regime targets, continuous future-return targets, and a joint objective over both, so representation learning is not locked to one label space.
- Multi-asset capable: the same trainer and study stack now handles both single-series and portfolio-style multi-series tasks.
- Research-friendly: the same core trainer powers single-run experiments, multi-seed benchmarks, and walk-forward validation.
- Backtest-friendly: trading runs now report risk-aware metrics instead of only raw reward and terminal equity.
- Comparison-ready: multiple experiment configs can be ranked into a single leaderboard without bespoke scripts.
- Constraint-aware: study and overnight search can optimize one metric while enforcing separate feasibility bounds.
- Multi-objective ready: study outputs can now surface Pareto-efficient configurations instead of forcing everything into one scalar score.
- Matrix-ready: one matrix spec can compare methods across multiple tasks without bespoke orchestration scripts.
- Weighted-benchmark ready: matrix leaderboards can now prioritize important tasks through per-task weights instead of forcing equal contribution.
- Real-data ready: historical CSVs can be loaded, sorted, filtered, and split by explicit time boundaries.
- Long-run friendly: the optimizer can keep searching unattended until a hard deadline or `STOP` file while persisting active-generation state for crash recovery.
- Open-source ready: packaging, tests, configs, and license are included.

The first baseline remains intentionally simple: a linear actor-critic agent over encoded time-series windows.
That keeps the framework lightweight while leaving explicit extension points for deep sequence models later.
