from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from tsrl_lite.benchmark import run_benchmark, run_walk_forward_benchmark
from tsrl_lite.config import load_experiment_config
from tsrl_lite.trainer import train_experiment
from tsrl_lite.utils import dump_json, dump_records_csv, ensure_dir


@dataclass(slots=True)
class StudyArtifacts:
    root_dir: Path
    summary_path: Path
    leaderboard_csv_path: Path
    leaderboard_md_path: Path
    metrics_csv_path: Path | None = None
    metrics_md_path: Path | None = None
    pareto_frontier_path: Path | None = None
    pareto_frontier_md_path: Path | None = None
    resolved_config_dir: Path | None = None


def slugify_name(value: str) -> str:
    output: list[str] = []
    prev_sep = False
    for char in value.lower():
        if char.isalnum():
            output.append(char)
            prev_sep = False
            continue
        if not prev_sep:
            output.append("_")
            prev_sep = True
    slug = "".join(output).strip("_")
    return slug or "experiment"


def flatten_metric_dict(metrics: dict[str, object], prefix: str) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            flattened.update(flatten_metric_dict(value, prefix=f"{prefix}{key}_"))
            continue
        if isinstance(value, int | float) and not isinstance(value, bool):
            flattened[f"{prefix}{key}"] = float(value)
    return flattened


def add_numeric_metric_catalog(
    metric_catalog: dict[str, float],
    metrics: dict[str, object],
    *,
    prefix: str,
    include_unprefixed: bool = False,
) -> None:
    for metric_name, metric_value in metrics.items():
        if not isinstance(metric_value, int | float) or isinstance(metric_value, bool):
            continue
        numeric_value = float(metric_value)
        metric_catalog[f"{prefix}{metric_name}"] = numeric_value
        if include_unprefixed:
            metric_catalog.setdefault(str(metric_name), numeric_value)


def add_aggregate_metric_catalog(
    metric_catalog: dict[str, float],
    metrics: dict[str, object],
    *,
    prefix: str,
    include_unprefixed: bool = False,
) -> None:
    for metric_name, metric_payload in metrics.items():
        if not isinstance(metric_payload, dict):
            continue
        mean_value: float | None = None
        for stat_name, stat_value in metric_payload.items():
            if not isinstance(stat_value, int | float) or isinstance(stat_value, bool):
                continue
            numeric_value = float(stat_value)
            metric_catalog[f"{prefix}{metric_name}.{stat_name}"] = numeric_value
            if stat_name == "mean":
                mean_value = numeric_value
        if mean_value is None:
            continue
        metric_catalog[f"{prefix}{metric_name}"] = mean_value
        if include_unprefixed:
            metric_catalog.setdefault(str(metric_name), mean_value)


def build_metric_catalog(
    mode: str,
    summary: dict[str, object],
    evaluation_metrics: dict[str, object],
    validation_metrics: dict[str, object],
) -> dict[str, float]:
    metric_catalog: dict[str, float] = {}
    if mode == "train":
        add_numeric_metric_catalog(metric_catalog, evaluation_metrics, prefix="evaluation.", include_unprefixed=True)
        add_numeric_metric_catalog(metric_catalog, validation_metrics, prefix="validation.")
        train_update_metrics = summary.get("train_update_metrics")
        if isinstance(train_update_metrics, dict):
            overall_metrics = train_update_metrics.get("overall")
            if isinstance(overall_metrics, dict):
                add_aggregate_metric_catalog(metric_catalog, overall_metrics, prefix="train.")
            tail_metrics = train_update_metrics.get("tail")
            if isinstance(tail_metrics, dict):
                add_aggregate_metric_catalog(metric_catalog, tail_metrics, prefix="train_tail.")
        return metric_catalog

    add_aggregate_metric_catalog(metric_catalog, evaluation_metrics, prefix="evaluation.", include_unprefixed=True)
    add_aggregate_metric_catalog(metric_catalog, validation_metrics, prefix="validation.")
    aggregate_train_update = summary.get("aggregate_train_update")
    if isinstance(aggregate_train_update, dict):
        add_aggregate_metric_catalog(metric_catalog, aggregate_train_update, prefix="train.")
    aggregate_train_update_tail = summary.get("aggregate_train_update_tail")
    if isinstance(aggregate_train_update_tail, dict):
        add_aggregate_metric_catalog(metric_catalog, aggregate_train_update_tail, prefix="train_tail.")
    return metric_catalog


def _report_column_name(metric_name: str) -> str:
    return f"report_{slugify_name(metric_name)}"


def normalize_metric_directions(
    raw_directions: object,
    field_name: str,
) -> dict[str, str]:
    if raw_directions in (None, {}):
        return {}
    if not isinstance(raw_directions, dict):
        raise ValueError(f"{field_name} must be an object mapping metric names to 'max' or 'min'")

    normalized: dict[str, str] = {}
    for metric_name, direction in raw_directions.items():
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError(f"{field_name} metric names must be non-empty strings")
        if direction not in {"max", "min"}:
            raise ValueError(f"{field_name} '{metric_name}' must be 'max' or 'min'")
        normalized[metric_name.strip()] = str(direction)
    return normalized


def normalize_metric_name_list(
    raw_metrics: object,
    field_name: str,
) -> list[str]:
    if raw_metrics in (None, []):
        return []
    if not isinstance(raw_metrics, list) or any(not isinstance(metric, str) for metric in raw_metrics):
        raise ValueError(f"{field_name} must be a list of metric names")

    normalized: list[str] = []
    for metric_name in raw_metrics:
        stripped = metric_name.strip()
        if not stripped:
            raise ValueError(f"{field_name} metric names must be non-empty strings")
        if stripped not in normalized:
            normalized.append(stripped)
    return normalized


def normalize_metric_constraints(
    raw_constraints: object,
) -> dict[str, dict[str, float]]:
    if raw_constraints in (None, {}):
        return {}
    if not isinstance(raw_constraints, dict):
        raise ValueError("metric_constraints must be an object mapping metric names to bounds")

    normalized: dict[str, dict[str, float]] = {}
    for metric_name, bounds in raw_constraints.items():
        if not isinstance(metric_name, str) or not metric_name.strip():
            raise ValueError("metric constraint names must be non-empty strings")
        if not isinstance(bounds, dict):
            raise ValueError(f"metric constraint '{metric_name}' must be an object")
        if "min" not in bounds and "max" not in bounds:
            raise ValueError(f"metric constraint '{metric_name}' must include 'min' and/or 'max'")

        normalized_bounds: dict[str, float] = {}
        for key in ("min", "max"):
            if key not in bounds:
                continue
            value = bounds[key]
            if not isinstance(value, int | float) or isinstance(value, bool):
                raise ValueError(f"metric constraint '{metric_name}.{key}' must be numeric")
            normalized_bounds[key] = float(value)

        if "min" in normalized_bounds and "max" in normalized_bounds:
            if normalized_bounds["min"] > normalized_bounds["max"]:
                raise ValueError(f"metric constraint '{metric_name}' has min > max")

        normalized[metric_name.strip()] = normalized_bounds
    return normalized


def extract_metric_value(
    metric_name: str,
    metric_catalog: dict[str, float],
) -> float:
    if metric_name not in metric_catalog:
        raise KeyError(f"metric '{metric_name}' missing from available study metrics")
    return float(metric_catalog[metric_name])


def prepare_metric_report_names(
    selection_metric: str,
    metric_constraints: dict[str, dict[str, float]],
    report_metrics: list[str] | None,
    pareto_metrics: dict[str, str],
) -> list[str]:
    ordered_metrics: list[str] = []

    def add_metric(metric_name: str) -> None:
        if metric_name not in ordered_metrics:
            ordered_metrics.append(metric_name)

    add_metric(selection_metric)
    for metric_name in metric_constraints:
        add_metric(metric_name)
    for metric_name in report_metrics or []:
        add_metric(metric_name)
    for metric_name in pareto_metrics:
        add_metric(metric_name)
    return ordered_metrics


def build_requested_metric_values(
    metric_catalog: dict[str, float],
    metric_names: list[str],
) -> dict[str, float]:
    return {
        metric_name: extract_metric_value(metric_name, metric_catalog)
        for metric_name in metric_names
    }


def evaluate_metric_constraints(
    metric_catalog: dict[str, float],
    metric_constraints: dict[str, dict[str, float]],
) -> tuple[bool, float, list[str]]:
    if not metric_constraints:
        return True, 0.0, []

    feasible = True
    violation_score = 0.0
    violations: list[str] = []
    for metric_name, bounds in metric_constraints.items():
        value = extract_metric_value(metric_name, metric_catalog)
        if "min" in bounds and value < bounds["min"]:
            feasible = False
            gap = bounds["min"] - value
            violation_score += gap / max(abs(bounds["min"]), 1.0)
            violations.append(f"{metric_name}<{bounds['min']:.6f} (got {value:.6f})")
        if "max" in bounds and value > bounds["max"]:
            feasible = False
            gap = value - bounds["max"]
            violation_score += gap / max(abs(bounds["max"]), 1.0)
            violations.append(f"{metric_name}>{bounds['max']:.6f} (got {value:.6f})")
    return feasible, float(violation_score), violations


def rank_leaderboard(rows: list[dict[str, object]], selection_mode: str) -> list[dict[str, object]]:
    if selection_mode not in {"max", "min"}:
        raise ValueError("selection_mode must be 'max' or 'min'")

    def row_key(row: dict[str, object]) -> tuple[bool, float, float]:
        feasible = bool(row.get("constraint_feasible", True))
        violation_score = float(row.get("constraint_violation_score", 0.0))
        selection_value = float(row["selection_value"])
        selection_key = -selection_value if selection_mode == "max" else selection_value
        return (not feasible, violation_score, selection_key)

    sorted_rows = sorted(rows, key=row_key)
    ranked_rows: list[dict[str, object]] = []
    for rank, row in enumerate(sorted_rows, start=1):
        ranked_rows.append({"rank": rank, **row})
    return ranked_rows


def extract_selection_value(
    selection_metric: str,
    metric_catalog: dict[str, float],
) -> float:
    return extract_metric_value(selection_metric, metric_catalog)


def render_markdown_leaderboard(
    rows: list[dict[str, object]],
    selection_metric: str,
) -> str:
    header = [
        "| rank | experiment | env | encoder | agent | metric | value | feasible | violations | summary |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines = []
    for row in rows:
        feasible = "yes" if bool(row.get("constraint_feasible", True)) else "no"
        violations = int(row.get("constraint_violation_count", 0))
        lines.append(
            "| "
            f"{row['rank']} | "
            f"{row['experiment']} | "
            f"{row['env']} | "
            f"{row['encoder']} | "
            f"{row['agent']} | "
            f"{selection_metric} | "
            f"{float(row['selection_value']):.6f} | "
            f"{feasible} | "
            f"{violations} | "
            f"{row['summary_path']} |"
        )
    return "\n".join(header + lines) + "\n"


def render_markdown_metric_report(
    rows: list[dict[str, object]],
    metric_names: list[str],
) -> str:
    headers = ["rank", "experiment", "feasible", *metric_names]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, separator_line]
    for row in rows:
        values = [
            str(row["rank"]),
            str(row["experiment"]),
            "yes" if bool(row.get("constraint_feasible", True)) else "no",
        ]
        for metric_name in metric_names:
            metric_value = row.get(_report_column_name(metric_name))
            if metric_value is None:
                values.append("")
            else:
                values.append(f"{float(metric_value):.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _pareto_dominates(
    lhs: dict[str, object],
    rhs: dict[str, object],
    pareto_metrics: dict[str, str],
) -> bool:
    strictly_better = False
    for metric_name, direction in pareto_metrics.items():
        lhs_value = float(lhs[_report_column_name(metric_name)])
        rhs_value = float(rhs[_report_column_name(metric_name)])
        if direction == "max":
            if lhs_value < rhs_value:
                return False
            if lhs_value > rhs_value:
                strictly_better = True
            continue
        if lhs_value > rhs_value:
            return False
        if lhs_value < rhs_value:
            strictly_better = True
    return strictly_better


def compute_pareto_frontier(
    rows: list[dict[str, object]],
    pareto_metrics: dict[str, str],
) -> tuple[str, list[dict[str, object]]]:
    if not pareto_metrics:
        return "disabled", []

    feasible_rows = [row for row in rows if bool(row.get("constraint_feasible", True))]
    candidate_rows = feasible_rows if feasible_rows else list(rows)
    pool = "feasible" if feasible_rows else "all"
    frontier: list[dict[str, object]] = []

    for row in candidate_rows:
        dominated = False
        for other in candidate_rows:
            if other is row:
                continue
            if _pareto_dominates(other, row, pareto_metrics=pareto_metrics):
                dominated = True
                break
        if dominated:
            continue
        frontier.append(row)

    return pool, frontier


def render_markdown_pareto_frontier(
    rows: list[dict[str, object]],
    pareto_metrics: dict[str, str],
    pool: str,
) -> str:
    if not pareto_metrics:
        return ""
    headers = ["experiment", "pool", *[f"{metric} ({direction})" for metric, direction in pareto_metrics.items()]]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, separator_line]
    for row in rows:
        values = [str(row["experiment"]), pool]
        for metric_name in pareto_metrics:
            values.append(f"{float(row[_report_column_name(metric_name)]):.6f}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _serialize_row_value(value: object) -> object:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int | float | str):
        return value
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value)


def _normalize_tags(raw_tags: object) -> list[str]:
    if raw_tags is None:
        return []
    if not isinstance(raw_tags, list) or any(not isinstance(tag, str) for tag in raw_tags):
        raise ValueError("study spec tags must be a list of strings")
    return sorted({tag.strip() for tag in raw_tags if tag.strip()})


def _matches_tag_filter(
    tags: list[str],
    include_tags: list[str] | None,
    exclude_tags: list[str] | None,
) -> bool:
    tag_set = set(tags)
    include_set = set(include_tags or [])
    exclude_set = set(exclude_tags or [])
    if include_set and not include_set.issubset(tag_set):
        return False
    if exclude_set and tag_set.intersection(exclude_set):
        return False
    return True


def _format_field_token(key: str) -> str:
    parts = [part for part in key.split(".") if part]
    if not parts:
        return "field"
    tail = parts[-2:] if len(parts) >= 2 else parts
    return slugify_name("_".join(tail))


def _format_value_token(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float | str):
        return slugify_name(str(value))
    if isinstance(value, list):
        return slugify_name("_".join(str(item) for item in value))
    if isinstance(value, dict):
        if "name" in value and isinstance(value["name"], str):
            return slugify_name(str(value["name"]))
        return slugify_name(json.dumps(value, sort_keys=True))
    return slugify_name(str(value))


def _merge_override_maps(
    base_overrides: dict[str, object],
    extra_overrides: dict[str, object],
) -> dict[str, object]:
    merged = copy.deepcopy(base_overrides)
    for key, value in extra_overrides.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(current, value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _expand_grid_variants(
    grid: dict[str, object],
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    if not grid:
        return [({}, {}, "")]

    ordered_items = list(grid.items())
    option_lists: list[list[object]] = []
    for key, values in ordered_items:
        if not isinstance(values, list) or not values:
            raise ValueError(f"study spec grid '{key}' must be a non-empty list")
        option_lists.append(values)

    variants: list[tuple[dict[str, object], dict[str, object], str]] = []
    for combo in product(*option_lists):
        overrides: dict[str, object] = {}
        sweep_values: dict[str, object] = {}
        suffix_parts: list[str] = []
        for (key, _), value in zip(ordered_items, combo, strict=True):
            overrides[key] = copy.deepcopy(value)
            sweep_values[key] = copy.deepcopy(value)
            suffix_parts.append(f"{_format_field_token(key)}_{_format_value_token(value)}")
        variants.append((overrides, sweep_values, "__".join(suffix_parts)))
    return variants


def _load_json_payload(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge_dict(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dict(current, value)
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _set_nested_override(payload: dict[str, object], dotted_key: str, value: object) -> None:
    parts = dotted_key.split(".")
    cursor = payload
    for part in parts[:-1]:
        current = cursor.get(part)
        if not isinstance(current, dict):
            current = {}
            cursor[part] = current
        cursor = current
    current = cursor.get(parts[-1])
    if isinstance(current, dict) and isinstance(value, dict):
        cursor[parts[-1]] = _deep_merge_dict(current, value)
        return
    cursor[parts[-1]] = copy.deepcopy(value)


def apply_config_overrides(
    payload: dict[str, object],
    overrides: dict[str, object],
) -> dict[str, object]:
    updated = copy.deepcopy(payload)
    for key, value in overrides.items():
        if "." in key:
            _set_nested_override(updated, key, value)
            continue
        current = updated.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            updated[key] = _deep_merge_dict(current, value)
            continue
        updated[key] = copy.deepcopy(value)
    return updated


def run_study(
    config_paths: list[str | Path],
    mode: str = "train",
    output_dir: str | Path | None = None,
    selection_metric: str = "mean_terminal_score",
    selection_mode: str = "max",
    metric_constraints: dict[str, dict[str, float]] | None = None,
    report_metrics: list[str] | None = None,
    pareto_metrics: dict[str, str] | None = None,
    seeds: list[int] | None = None,
    walk_forward_folds: int | None = None,
    train_ratio_start: float = 0.5,
) -> tuple[StudyArtifacts, dict]:
    if not config_paths:
        raise ValueError("config_paths must not be empty")
    if mode not in {"train", "benchmark", "walk-forward"}:
        raise ValueError("mode must be 'train', 'benchmark', or 'walk-forward'")
    if mode == "benchmark" and not seeds:
        raise ValueError("seeds are required for benchmark study mode")
    if mode == "walk-forward" and walk_forward_folds is None:
        raise ValueError("walk_forward_folds is required for walk-forward study mode")

    items = [{"config_path": Path(config_path), "row_fields": {}} for config_path in config_paths]
    return _run_study_items(
        study_items=items,
        mode=mode,
        output_dir=output_dir,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        metric_constraints=normalize_metric_constraints(metric_constraints),
        report_metrics=normalize_metric_name_list(report_metrics, field_name="report_metrics"),
        pareto_metrics=normalize_metric_directions(pareto_metrics, field_name="pareto_metrics"),
        seeds=seeds,
        walk_forward_folds=walk_forward_folds,
        train_ratio_start=train_ratio_start,
    )


def _run_study_items(
    study_items: list[dict[str, object]],
    mode: str,
    output_dir: str | Path | None,
    selection_metric: str,
    selection_mode: str,
    metric_constraints: dict[str, dict[str, float]],
    report_metrics: list[str] | None,
    pareto_metrics: dict[str, str],
    seeds: list[int] | None,
    walk_forward_folds: int | None,
    train_ratio_start: float,
) -> tuple[StudyArtifacts, dict]:
    study_root = ensure_dir(output_dir or Path("runs") / "study")
    rows: list[dict[str, object]] = []
    metric_report_names = prepare_metric_report_names(
        selection_metric=selection_metric,
        metric_constraints=metric_constraints,
        report_metrics=report_metrics,
        pareto_metrics=pareto_metrics,
    )

    for index, item in enumerate(study_items, start=1):
        rows.append(
            _evaluate_study_item(
                item=item,
                index=index,
                study_root=study_root,
                mode=mode,
                selection_metric=selection_metric,
                metric_constraints=metric_constraints,
                metric_report_names=metric_report_names,
                seeds=seeds,
                walk_forward_folds=walk_forward_folds,
                train_ratio_start=train_ratio_start,
            )
        )

    return _finalize_study_outputs(
        study_root=study_root,
        rows=rows,
        selection_metric=selection_metric,
        selection_mode=selection_mode,
        metric_constraints=metric_constraints,
        metric_report_names=metric_report_names,
        pareto_metrics=pareto_metrics,
        mode=mode,
        seeds=seeds,
        walk_forward_folds=walk_forward_folds,
        train_ratio_start=train_ratio_start,
        config_paths=[Path(item["config_path"]) for item in study_items],
    )


def _evaluate_study_item(
    item: dict[str, object],
    index: int,
    study_root: Path,
    mode: str,
    selection_metric: str,
    metric_constraints: dict[str, dict[str, float]],
    metric_report_names: list[str],
    seeds: list[int] | None,
    walk_forward_folds: int | None,
    train_ratio_start: float,
) -> dict[str, object]:
    path = Path(item["config_path"])
    config = load_experiment_config(path)
    run_output_dir = study_root / f"{index:02d}_{slugify_name(config.experiment_name)}"

    if mode == "train":
        artifacts, summary = train_experiment(path, output_dir=run_output_dir)
        evaluation_metrics = summary["evaluation"]
        validation_metrics = summary.get("best_validation") or {}
    elif mode == "benchmark":
        artifacts, summary = run_benchmark(path, seeds=seeds or [], output_dir=run_output_dir)
        evaluation_metrics = summary["aggregate_evaluation"]
        validation_metrics = summary.get("aggregate_best_validation") or {}
    else:
        artifacts, summary = run_walk_forward_benchmark(
            path,
            n_folds=int(walk_forward_folds or 0),
            train_ratio_start=train_ratio_start,
            output_dir=run_output_dir,
        )
        evaluation_metrics = summary["aggregate_evaluation"]
        validation_metrics = summary.get("aggregate_best_validation") or {}
    metric_catalog = build_metric_catalog(
        mode=mode,
        summary=summary,
        evaluation_metrics=evaluation_metrics,
        validation_metrics=validation_metrics,
    )

    row = {
        "experiment": config.experiment_name,
        "config_path": str(path),
        "mode": mode,
        "output_dir": str(run_output_dir),
        "summary_path": str(artifacts.summary_path),
        "env": config.env.name,
        "encoder": config.encoder.name,
        "agent": config.agent.name,
        "selection_metric": selection_metric,
        "selection_value": extract_selection_value(selection_metric, metric_catalog),
    }
    constraint_feasible, constraint_violation_score, constraint_violations = evaluate_metric_constraints(
        metric_catalog=metric_catalog,
        metric_constraints=metric_constraints,
    )
    row["constraint_feasible"] = constraint_feasible
    row["constraint_violation_score"] = float(constraint_violation_score)
    row["constraint_violation_count"] = int(len(constraint_violations))
    row["constraint_violations"] = constraint_violations
    requested_metric_values = build_requested_metric_values(
        metric_catalog=metric_catalog,
        metric_names=metric_report_names,
    )
    for metric_name, metric_value in requested_metric_values.items():
        row[_report_column_name(metric_name)] = float(metric_value)
    row.update(flatten_metric_dict(evaluation_metrics, prefix="evaluation_"))
    if validation_metrics:
        row.update(flatten_metric_dict(validation_metrics, prefix="validation_"))
    if mode == "train":
        train_update_metrics = summary.get("train_update_metrics")
        if isinstance(train_update_metrics, dict):
            overall_metrics = train_update_metrics.get("overall")
            if isinstance(overall_metrics, dict):
                row.update(flatten_metric_dict(overall_metrics, prefix="train_"))
            tail_metrics = train_update_metrics.get("tail")
            if isinstance(tail_metrics, dict):
                row.update(flatten_metric_dict(tail_metrics, prefix="train_tail_"))
    else:
        aggregate_train_update = summary.get("aggregate_train_update")
        if isinstance(aggregate_train_update, dict):
            row.update(flatten_metric_dict(aggregate_train_update, prefix="train_"))
        aggregate_train_update_tail = summary.get("aggregate_train_update_tail")
        if isinstance(aggregate_train_update_tail, dict):
            row.update(flatten_metric_dict(aggregate_train_update_tail, prefix="train_tail_"))
    if mode == "benchmark":
        row["benchmark_seeds"] = ",".join(str(seed) for seed in seeds or [])
    if mode == "walk-forward":
        row["walk_forward_folds"] = int(walk_forward_folds or 0)
        row["train_ratio_start"] = float(train_ratio_start)
    row.update(
        {
            key: _serialize_row_value(value)
            for key, value in dict(item.get("row_fields", {})).items()
        }
    )
    return row


def _finalize_study_outputs(
    study_root: Path,
    rows: list[dict[str, object]],
    selection_metric: str,
    selection_mode: str,
    metric_constraints: dict[str, dict[str, float]],
    metric_report_names: list[str],
    pareto_metrics: dict[str, str],
    mode: str,
    seeds: list[int] | None,
    walk_forward_folds: int | None,
    train_ratio_start: float,
    config_paths: list[Path],
) -> tuple[StudyArtifacts, dict]:
    ranked_rows = rank_leaderboard(rows, selection_mode=selection_mode)

    summary_path = study_root / "study_summary.json"
    leaderboard_csv_path = study_root / "leaderboard.csv"
    leaderboard_md_path = study_root / "leaderboard.md"
    metrics_csv_path = study_root / "metrics_report.csv" if metric_report_names else None
    metrics_md_path = study_root / "metrics_report.md" if metric_report_names else None
    pareto_frontier_path = study_root / "pareto_frontier.json" if pareto_metrics else None
    pareto_frontier_md_path = study_root / "pareto_frontier.md" if pareto_metrics else None
    leaderboard_md_path.write_text(
        render_markdown_leaderboard(ranked_rows, selection_metric=selection_metric),
        encoding="utf-8",
    )
    dump_records_csv(leaderboard_csv_path, ranked_rows)
    if metrics_csv_path is not None and metrics_md_path is not None:
        metric_rows = [
            {
                "rank": int(row["rank"]),
                "experiment": row["experiment"],
                "constraint_feasible": bool(row.get("constraint_feasible", True)),
                **{
                    _report_column_name(metric_name): float(row[_report_column_name(metric_name)])
                    for metric_name in metric_report_names
                },
            }
            for row in ranked_rows
        ]
        dump_records_csv(metrics_csv_path, metric_rows)
        metrics_md_path.write_text(
            render_markdown_metric_report(ranked_rows, metric_names=metric_report_names),
            encoding="utf-8",
        )
    feasible_rows = [row for row in ranked_rows if bool(row.get("constraint_feasible", True))]
    pareto_pool, pareto_frontier_rows = compute_pareto_frontier(ranked_rows, pareto_metrics=pareto_metrics)
    pareto_frontier_summary = [
        {
            "experiment": row["experiment"],
            "rank": int(row["rank"]),
            "constraint_feasible": bool(row.get("constraint_feasible", True)),
            "metrics": {
                metric_name: float(row[_report_column_name(metric_name)])
                for metric_name in pareto_metrics
            },
            "summary_path": row["summary_path"],
        }
        for row in pareto_frontier_rows
    ]
    if pareto_frontier_path is not None and pareto_frontier_md_path is not None:
        dump_json(
            pareto_frontier_path,
            {
                "pool": pareto_pool,
                "pareto_metrics": pareto_metrics,
                "frontier": pareto_frontier_summary,
            },
        )
        pareto_frontier_md_path.write_text(
            render_markdown_pareto_frontier(
                pareto_frontier_rows,
                pareto_metrics=pareto_metrics,
                pool=pareto_pool,
            ),
            encoding="utf-8",
        )
    summary = {
        "mode": mode,
        "selection_metric": selection_metric,
        "selection_mode": selection_mode,
        "metric_constraints": metric_constraints,
        "report_metrics": metric_report_names,
        "pareto_metrics": pareto_metrics,
        "pareto_pool": pareto_pool,
        "pareto_frontier": pareto_frontier_summary,
        "config_paths": [str(path) for path in config_paths],
        "benchmark_seeds": [int(seed) for seed in seeds or []],
        "walk_forward_folds": int(walk_forward_folds or 0),
        "train_ratio_start": float(train_ratio_start),
        "run_count": len(ranked_rows),
        "top_experiment": ranked_rows[0]["experiment"],
        "top_feasible_experiment": feasible_rows[0]["experiment"] if feasible_rows else None,
        "feasible_run_count": int(len(feasible_rows)),
        "infeasible_run_count": int(len(ranked_rows) - len(feasible_rows)),
        "leaderboard": ranked_rows,
    }
    dump_json(summary_path, summary)
    artifacts = StudyArtifacts(
        root_dir=study_root,
        summary_path=summary_path,
        leaderboard_csv_path=leaderboard_csv_path,
        leaderboard_md_path=leaderboard_md_path,
        metrics_csv_path=metrics_csv_path,
        metrics_md_path=metrics_md_path,
        pareto_frontier_path=pareto_frontier_path,
        pareto_frontier_md_path=pareto_frontier_md_path,
    )
    return artifacts, summary


def run_study_spec(
    spec_path: str | Path,
    output_dir: str | Path | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
) -> tuple[StudyArtifacts, dict]:
    spec_file = Path(spec_path)
    spec = _load_json_payload(spec_file)
    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("study spec must contain a non-empty 'experiments' list")

    study_root = ensure_dir(output_dir or Path("runs") / f"study_{slugify_name(spec_file.stem)}")
    resolved_config_dir = ensure_dir(study_root / "resolved_configs")
    study_items: list[dict[str, object]] = []
    global_tags = _normalize_tags(spec.get("tags"))
    include_tags = _normalize_tags(include_tags)
    exclude_tags = _normalize_tags(exclude_tags)
    generated_variants = 0

    base_config_path = spec.get("base_config")
    for experiment in experiments:
        if not isinstance(experiment, dict):
            raise ValueError("each study spec experiment must be an object")

        config_ref = experiment.get("config", base_config_path)
        if not isinstance(config_ref, str):
            raise ValueError("each study spec experiment needs a string 'config' or spec-level 'base_config'")

        config_path = (spec_file.parent / config_ref).resolve()
        payload = _load_json_payload(config_path)
        base_overrides = experiment.get("overrides", {})
        if not isinstance(base_overrides, dict):
            raise ValueError("experiment overrides must be an object")
        grid = experiment.get("grid", {})
        if not isinstance(grid, dict):
            raise ValueError("experiment grid must be an object")

        experiment_name = experiment.get("name")
        default_name = payload.get("experiment_name", "experiment")
        if not isinstance(default_name, str):
            default_name = "experiment"
        base_name = experiment_name if isinstance(experiment_name, str) else default_name
        experiment_tags = sorted({*global_tags, *_normalize_tags(experiment.get("tags"))})

        for variant_overrides, sweep_values, suffix in _expand_grid_variants(grid):
            generated_variants += 1
            combined_overrides = _merge_override_maps(base_overrides, variant_overrides)
            variant_name = base_name if not suffix else f"{base_name}__{suffix}"
            variant_tags = sorted(
                {
                    *experiment_tags,
                    *(f"sweep:{key}" for key in sweep_values),
                }
            )
            if not _matches_tag_filter(variant_tags, include_tags=include_tags, exclude_tags=exclude_tags):
                continue

            resolved_payload = apply_config_overrides(payload, combined_overrides)
            if "experiment_name" not in combined_overrides:
                resolved_payload["experiment_name"] = variant_name

            resolved_index = len(study_items) + 1
            resolved_path = resolved_config_dir / f"{resolved_index:02d}_{slugify_name(variant_name)}.json"
            dump_json(resolved_path, resolved_payload)

            row_fields: dict[str, object] = {
                "study_tags": variant_tags,
                "source_config": str(config_path),
                "source_experiment": base_name,
            }
            for key, value in sweep_values.items():
                row_fields[f"sweep_{slugify_name(key.replace('.', '_'))}"] = value
            study_items.append({"config_path": resolved_path, "row_fields": row_fields})

    if not study_items:
        raise ValueError("study spec did not produce any experiments after tag filtering")

    artifacts, summary = _run_study_items(
        study_items=study_items,
        mode=str(spec.get("mode", "train")),
        output_dir=study_root,
        selection_metric=str(spec.get("selection_metric", "mean_terminal_score")),
        selection_mode=str(spec.get("selection_mode", "max")),
        metric_constraints=normalize_metric_constraints(spec.get("metric_constraints")),
        report_metrics=normalize_metric_name_list(spec.get("report_metrics"), field_name="report_metrics"),
        pareto_metrics=normalize_metric_directions(spec.get("pareto_metrics"), field_name="pareto_metrics"),
        seeds=spec.get("benchmark_seeds"),
        walk_forward_folds=spec.get("walk_forward_folds"),
        train_ratio_start=float(spec.get("train_ratio_start", 0.5)),
    )

    summary["spec_path"] = str(spec_file)
    summary["resolved_config_dir"] = str(resolved_config_dir)
    summary["include_tags"] = include_tags
    summary["exclude_tags"] = exclude_tags
    summary["generated_variants"] = int(generated_variants)
    dump_json(artifacts.summary_path, summary)
    artifacts = StudyArtifacts(
        root_dir=artifacts.root_dir,
        summary_path=artifacts.summary_path,
        leaderboard_csv_path=artifacts.leaderboard_csv_path,
        leaderboard_md_path=artifacts.leaderboard_md_path,
        metrics_csv_path=artifacts.metrics_csv_path,
        metrics_md_path=artifacts.metrics_md_path,
        pareto_frontier_path=artifacts.pareto_frontier_path,
        pareto_frontier_md_path=artifacts.pareto_frontier_md_path,
        resolved_config_dir=resolved_config_dir,
    )
    return artifacts, summary
