from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from tsrl_lite.study import (
    StudyArtifacts,
    _report_column_name,
    _expand_grid_variants,
    _load_json_payload,
    _merge_override_maps,
    apply_config_overrides,
    normalize_metric_constraints,
    run_study_spec,
    slugify_name,
)
from tsrl_lite.utils import dump_json, dump_records_csv, ensure_dir


@dataclass(slots=True)
class MatrixArtifacts:
    root_dir: Path
    summary_path: Path
    leaderboard_csv_path: Path
    leaderboard_md_path: Path
    task_matrix_csv_path: Path
    task_matrix_md_path: Path
    pairwise_csv_path: Path
    pairwise_md_path: Path
    generated_spec_dir: Path
    resolved_task_config_dir: Path
    task_output_dir: Path
    method_metrics_csv_path: Path | None = None
    method_metrics_md_path: Path | None = None


def _compute_payload_digest(payload: dict[str, object]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _normalize_name(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _normalize_tag_list(raw_tags: object, field_name: str) -> list[str]:
    if raw_tags is None:
        return []
    if not isinstance(raw_tags, list) or any(not isinstance(tag, str) for tag in raw_tags):
        raise ValueError(f"{field_name} must be a list of strings")
    return sorted({tag.strip() for tag in raw_tags if tag.strip()})


def _normalize_task_weight(raw_weight: object, field_name: str) -> float:
    if not isinstance(raw_weight, int | float) or isinstance(raw_weight, bool):
        raise ValueError(f"{field_name} must be numeric")
    weight = float(raw_weight)
    if weight <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    return weight


def _merge_tags(*tag_groups: list[str]) -> list[str]:
    merged: set[str] = set()
    for group in tag_groups:
        merged.update(tag for tag in group if tag)
    return sorted(merged)


def _build_task_variant_name(base_name: str, suffix: str) -> str:
    return base_name if not suffix else f"{base_name}__{suffix}"


def _build_generated_method_specs(
    methods: list[dict[str, object]],
    task_name: str,
    task_tags: list[str],
    global_tags: list[str],
) -> list[dict[str, object]]:
    generated_methods: list[dict[str, object]] = []
    seen_method_names: set[str] = set()
    for method in methods:
        method_name = _normalize_name(method.get("name"), field_name="matrix method name")
        if method_name in seen_method_names:
            raise ValueError(f"matrix methods must have unique names, got duplicate '{method_name}'")
        seen_method_names.add(method_name)
        if "config" in method:
            raise ValueError("matrix methods must not set 'config'; use overrides and grid on top of task configs")

        method_tags = _normalize_tag_list(method.get("tags"), field_name=f"matrix method '{method_name}' tags")
        overrides = method.get("overrides", {})
        if not isinstance(overrides, dict):
            raise ValueError(f"matrix method '{method_name}' overrides must be an object")
        grid = method.get("grid", {})
        if not isinstance(grid, dict):
            raise ValueError(f"matrix method '{method_name}' grid must be an object")

        generated_method = {
            "name": method_name,
            "tags": _merge_tags(
                global_tags,
                task_tags,
                method_tags,
                [f"task:{task_name}", f"method:{method_name}"],
            ),
        }
        if overrides:
            generated_method["overrides"] = copy.deepcopy(overrides)
        if grid:
            generated_method["grid"] = copy.deepcopy(grid)
        generated_methods.append(generated_method)
    return generated_methods


def _resolve_task_setting(
    task: dict[str, object],
    spec: dict[str, object],
    key: str,
    default: object,
) -> object:
    return copy.deepcopy(task.get(key, spec.get(key, default)))


def _build_task_summary_record(
    task_name: str,
    task_tags: list[str],
    task_weight: float,
    task_status: str,
    study_artifacts: StudyArtifacts,
    study_summary: dict[str, object],
) -> dict[str, object]:
    return {
        "task": task_name,
        "tags": task_tags,
        "task_weight": float(task_weight),
        "task_status": task_status,
        "mode": study_summary["mode"],
        "selection_metric": study_summary["selection_metric"],
        "selection_mode": study_summary["selection_mode"],
        "summary_path": str(study_artifacts.summary_path),
        "leaderboard_csv_path": str(study_artifacts.leaderboard_csv_path),
        "leaderboard_md_path": str(study_artifacts.leaderboard_md_path),
        "metrics_csv_path": str(study_artifacts.metrics_csv_path) if study_artifacts.metrics_csv_path else None,
        "metrics_md_path": str(study_artifacts.metrics_md_path) if study_artifacts.metrics_md_path else None,
        "pareto_frontier_path": (
            str(study_artifacts.pareto_frontier_path) if study_artifacts.pareto_frontier_path else None
        ),
        "pareto_frontier_md_path": (
            str(study_artifacts.pareto_frontier_md_path) if study_artifacts.pareto_frontier_md_path else None
        ),
        "run_count": int(study_summary["run_count"]),
        "top_experiment": study_summary["top_experiment"],
        "top_feasible_experiment": study_summary["top_feasible_experiment"],
        "feasible_run_count": int(study_summary["feasible_run_count"]),
        "infeasible_run_count": int(study_summary["infeasible_run_count"]),
    }


def _build_method_leaderboard(
    task_results: list[dict[str, object]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not task_results:
        return [], []

    task_names = [str(task_result["task"]) for task_result in task_results]
    task_weights = {
        str(task_result["task"]): float(task_result["task_weight"])
        for task_result in task_results
    }
    task_tokens = {task_name: slugify_name(task_name) for task_name in task_names}
    total_tasks = len(task_results)
    total_task_weight = sum(task_weights.values())
    method_breakdown: dict[str, dict[str, dict[str, object]]] = {}

    for task_result in task_results:
        task_name = str(task_result["task"])
        summary = dict(task_result["summary"])
        leaderboard = list(summary["leaderboard"])
        task_run_count = len(leaderboard)
        selection_metric = str(summary["selection_metric"])
        for row in leaderboard:
            rank = int(row["rank"])
            normalized_rank_score = 1.0 if task_run_count <= 1 else float(task_run_count - rank) / float(
                task_run_count - 1
            )
            method_breakdown.setdefault(str(row["experiment"]), {})[task_name] = {
                "rank": rank,
                "selection_value": float(row["selection_value"]),
                "selection_metric": selection_metric,
                "constraint_feasible": bool(row.get("constraint_feasible", True)),
                "normalized_rank_score": float(normalized_rank_score),
                "summary_path": str(row["summary_path"]),
                "task_weight": task_weights[task_name],
            }

    leaderboard_rows: list[dict[str, object]] = []
    task_matrix_rows: list[dict[str, object]] = []
    for method_name, per_task in method_breakdown.items():
        tasks_covered = len(per_task)
        feasible_task_count = sum(1 for payload in per_task.values() if bool(payload["constraint_feasible"]))
        task_weight_total = sum(float(payload["task_weight"]) for payload in per_task.values())
        feasible_task_weight = sum(
            float(payload["task_weight"]) for payload in per_task.values() if bool(payload["constraint_feasible"])
        )
        wins = sum(1 for payload in per_task.values() if int(payload["rank"]) == 1)
        mean_rank = sum(
            float(payload["rank"]) * float(payload["task_weight"])
            for payload in per_task.values()
        ) / float(task_weight_total)
        mean_normalized_rank_score = sum(
            float(payload["normalized_rank_score"]) * float(payload["task_weight"])
            for payload in per_task.values()
        ) / float(task_weight_total)
        coverage_ratio = float(tasks_covered) / float(total_tasks)
        weighted_coverage_ratio = float(task_weight_total) / float(total_task_weight)
        feasible_task_ratio = float(feasible_task_count) / float(tasks_covered)
        feasible_task_weight_ratio = float(feasible_task_weight) / float(task_weight_total)

        leaderboard_row = {
            "method": method_name,
            "tasks_covered": int(tasks_covered),
            "task_coverage_ratio": float(coverage_ratio),
            "task_weight_total": float(task_weight_total),
            "weighted_task_coverage_ratio": float(weighted_coverage_ratio),
            "feasible_task_count": int(feasible_task_count),
            "feasible_task_ratio": float(feasible_task_ratio),
            "feasible_task_weight": float(feasible_task_weight),
            "feasible_task_weight_ratio": float(feasible_task_weight_ratio),
            "wins": int(wins),
            "mean_rank": float(mean_rank),
            "mean_normalized_rank_score": float(mean_normalized_rank_score),
        }
        task_matrix_row = dict(leaderboard_row)
        for task_name in task_names:
            token = task_tokens[task_name]
            task_payload = per_task.get(task_name)
            if task_payload is None:
                task_matrix_row[f"{token}_task_weight"] = float(task_weights[task_name])
                task_matrix_row[f"{token}_rank"] = None
                task_matrix_row[f"{token}_normalized_rank_score"] = None
                task_matrix_row[f"{token}_selection_metric"] = None
                task_matrix_row[f"{token}_selection_value"] = None
                task_matrix_row[f"{token}_constraint_feasible"] = None
                continue
            task_matrix_row[f"{token}_task_weight"] = float(task_payload["task_weight"])
            task_matrix_row[f"{token}_rank"] = int(task_payload["rank"])
            task_matrix_row[f"{token}_normalized_rank_score"] = float(task_payload["normalized_rank_score"])
            task_matrix_row[f"{token}_selection_metric"] = str(task_payload["selection_metric"])
            task_matrix_row[f"{token}_selection_value"] = float(task_payload["selection_value"])
            task_matrix_row[f"{token}_constraint_feasible"] = bool(task_payload["constraint_feasible"])
        leaderboard_rows.append(leaderboard_row)
        task_matrix_rows.append(task_matrix_row)

    leaderboard_rows = sorted(
        leaderboard_rows,
        key=lambda row: (
            -float(row["task_weight_total"]),
            -float(row["feasible_task_weight"]),
            -float(row["mean_normalized_rank_score"]),
            float(row["mean_rank"]),
            -int(row["wins"]),
            str(row["method"]),
        ),
    )
    ranked_leaderboard: list[dict[str, object]] = []
    ranked_matrix_rows: list[dict[str, object]] = []
    method_rank_map: dict[str, int] = {}
    for rank, row in enumerate(leaderboard_rows, start=1):
        ranked_row = {"rank": rank, **row}
        ranked_leaderboard.append(ranked_row)
        method_rank_map[str(row["method"])] = rank
    for row in task_matrix_rows:
        ranked_matrix_rows.append({"rank": method_rank_map[str(row["method"])], **row})
    ranked_matrix_rows = sorted(ranked_matrix_rows, key=lambda row: (int(row["rank"]), str(row["method"])))
    return ranked_leaderboard, ranked_matrix_rows


def _render_markdown_matrix_leaderboard(rows: list[dict[str, object]]) -> str:
    show_matrix_selection = any("matrix_selection_metric" in row for row in rows)
    if show_matrix_selection:
        header = [
            "| rank | method | tasks | task weight | feasible weight | wins | weighted rank | weighted score | matrix metric | value | feasible | violations |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    else:
        header = [
            "| rank | method | tasks | task weight | feasible weight | wins | weighted rank | weighted score |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    lines = []
    for row in rows:
        values = [
            str(row["rank"]),
            str(row["method"]),
            str(int(row["tasks_covered"])),
            f"{float(row['task_weight_total']):.3f}",
            f"{float(row['feasible_task_weight']):.3f}",
            str(int(row["wins"])),
            f"{float(row['mean_rank']):.3f}",
            f"{float(row['mean_normalized_rank_score']):.6f}",
        ]
        if show_matrix_selection:
            values.extend(
                [
                    str(row.get("matrix_selection_metric", "")),
                    f"{float(row['matrix_selection_value']):.6f}" if "matrix_selection_value" in row else "",
                    "yes" if bool(row.get("matrix_constraint_feasible", True)) else "no",
                    str(int(row.get("matrix_constraint_violation_count", 0))),
                ]
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(header + lines) + "\n"


def _render_markdown_task_matrix(
    rows: list[dict[str, object]],
    task_names: list[str],
    task_weights: dict[str, float],
) -> str:
    headers = [
        "rank",
        "method",
        *[f"{task_name} (w={task_weights[task_name]:.2f})" for task_name in task_names],
    ]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, separator_line]
    for row in rows:
        values = [str(row["rank"]), str(row["method"])]
        for task_name in task_names:
            token = slugify_name(task_name)
            rank_value = row.get(f"{token}_rank")
            if rank_value is None:
                values.append("")
                continue
            feasible = "yes" if bool(row.get(f"{token}_constraint_feasible")) else "no"
            score = float(row[f"{token}_normalized_rank_score"])
            values.append(f"r{int(rank_value)} / {score:.3f} / {feasible}")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _build_pairwise_comparisons(
    task_results: list[dict[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    method_names = sorted(
        {
            str(row["experiment"])
            for task_result in task_results
            for row in task_result["summary"]["leaderboard"]
        }
    )
    pairwise_rows: list[dict[str, object]] = []
    for method_name in method_names:
        for opponent_name in method_names:
            if method_name == opponent_name:
                continue
            wins = 0
            losses = 0
            ties = 0
            weighted_wins = 0.0
            weighted_losses = 0.0
            weighted_ties = 0.0
            compared_tasks = 0
            total_weight = 0.0
            for task_result in task_results:
                task_weight = float(task_result["task_weight"])
                rank_map = {
                    str(row["experiment"]): int(row["rank"])
                    for row in task_result["summary"]["leaderboard"]
                }
                if method_name not in rank_map or opponent_name not in rank_map:
                    continue
                compared_tasks += 1
                total_weight += task_weight
                if rank_map[method_name] < rank_map[opponent_name]:
                    wins += 1
                    weighted_wins += task_weight
                elif rank_map[method_name] > rank_map[opponent_name]:
                    losses += 1
                    weighted_losses += task_weight
                else:
                    ties += 1
                    weighted_ties += task_weight
            pairwise_rows.append(
                {
                    "method": method_name,
                    "opponent": opponent_name,
                    "tasks_compared": int(compared_tasks),
                    "task_weight_total": float(total_weight),
                    "wins": int(wins),
                    "losses": int(losses),
                    "ties": int(ties),
                    "weighted_wins": float(weighted_wins),
                    "weighted_losses": float(weighted_losses),
                    "weighted_ties": float(weighted_ties),
                    "weighted_net": float(weighted_wins - weighted_losses),
                    "weighted_win_rate": (float(weighted_wins / total_weight) if total_weight > 0.0 else None),
                }
            )
    return method_names, pairwise_rows


def _matrix_metric_column_name(metric_name: str, suffix: str) -> str:
    return f"metric_{slugify_name(metric_name)}_{suffix}"


def _extract_numeric(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _collect_matrix_report_metrics(task_results: list[dict[str, object]]) -> list[str]:
    metric_names: list[str] = []
    seen: set[str] = set()
    for task_result in task_results:
        summary = dict(task_result["summary"])
        raw_metrics = summary.get("report_metrics", [])
        if not isinstance(raw_metrics, list):
            continue
        for metric_name in raw_metrics:
            if not isinstance(metric_name, str) or not metric_name.strip():
                continue
            normalized = metric_name.strip()
            if normalized in seen:
                continue
            seen.add(normalized)
            metric_names.append(normalized)
    return metric_names


def _build_method_metric_report(
    task_results: list[dict[str, object]],
    method_leaderboard: list[dict[str, object]],
) -> tuple[list[str], list[dict[str, object]]]:
    metric_names = _collect_matrix_report_metrics(task_results)
    if not metric_names:
        return [], []

    total_task_weight = sum(float(task_result["task_weight"]) for task_result in task_results)
    metric_accumulators: dict[str, dict[str, dict[str, float]]] = {}

    for task_result in task_results:
        summary = dict(task_result["summary"])
        leaderboard = list(summary["leaderboard"])
        task_weight = float(task_result["task_weight"])
        report_metrics = [
            metric_name
            for metric_name in summary.get("report_metrics", [])
            if isinstance(metric_name, str) and metric_name.strip()
        ]
        for row in leaderboard:
            method_name = str(row["experiment"])
            method_payload = metric_accumulators.setdefault(method_name, {})
            for metric_name in report_metrics:
                value = row.get(_report_column_name(metric_name))
                if not isinstance(value, int | float) or isinstance(value, bool):
                    continue
                numeric_value = float(value)
                metric_payload = method_payload.setdefault(
                    metric_name,
                    {
                        "weighted_sum": 0.0,
                        "task_weight": 0.0,
                        "task_count": 0.0,
                        "min": numeric_value,
                        "max": numeric_value,
                    },
                )
                metric_payload["weighted_sum"] += numeric_value * task_weight
                metric_payload["task_weight"] += task_weight
                metric_payload["task_count"] += 1.0
                metric_payload["min"] = min(float(metric_payload["min"]), numeric_value)
                metric_payload["max"] = max(float(metric_payload["max"]), numeric_value)

    rows: list[dict[str, object]] = []
    for leaderboard_row in method_leaderboard:
        method_name = str(leaderboard_row["method"])
        row: dict[str, object] = {
            "rank": int(leaderboard_row["rank"]),
            "method": method_name,
        }
        for metric_name in metric_names:
            token_mean = _matrix_metric_column_name(metric_name, "weighted_mean")
            token_weight = _matrix_metric_column_name(metric_name, "task_weight")
            token_coverage = _matrix_metric_column_name(metric_name, "weighted_coverage_ratio")
            token_count = _matrix_metric_column_name(metric_name, "task_count")
            token_min = _matrix_metric_column_name(metric_name, "min")
            token_max = _matrix_metric_column_name(metric_name, "max")
            metric_payload = metric_accumulators.get(method_name, {}).get(metric_name)
            if metric_payload is None:
                row[token_mean] = None
                row[token_weight] = 0.0
                row[token_coverage] = 0.0
                row[token_count] = 0
                row[token_min] = None
                row[token_max] = None
                continue
            metric_weight = float(metric_payload["task_weight"])
            row[token_mean] = float(metric_payload["weighted_sum"]) / metric_weight
            row[token_weight] = metric_weight
            row[token_coverage] = metric_weight / float(total_task_weight) if total_task_weight > 0.0 else 0.0
            row[token_count] = int(metric_payload["task_count"])
            row[token_min] = float(metric_payload["min"])
            row[token_max] = float(metric_payload["max"])
        rows.append(row)
    return metric_names, rows


def _build_method_metric_catalog(
    leaderboard_row: dict[str, object],
    method_metric_row: dict[str, object] | None,
    metric_names: list[str],
) -> dict[str, float]:
    metric_catalog: dict[str, float] = {}
    for key, value in leaderboard_row.items():
        numeric_value = _extract_numeric(value)
        if numeric_value is None:
            continue
        metric_catalog[f"matrix.{key}"] = numeric_value

    if method_metric_row is None:
        return metric_catalog

    for metric_name in metric_names:
        weighted_mean = _extract_numeric(method_metric_row.get(_matrix_metric_column_name(metric_name, "weighted_mean")))
        if weighted_mean is not None:
            metric_catalog[metric_name] = weighted_mean
            metric_catalog[f"report.{metric_name}"] = weighted_mean

        task_weight = _extract_numeric(method_metric_row.get(_matrix_metric_column_name(metric_name, "task_weight")))
        if task_weight is not None:
            metric_catalog[f"report_task_weight.{metric_name}"] = task_weight

        coverage_ratio = _extract_numeric(
            method_metric_row.get(_matrix_metric_column_name(metric_name, "weighted_coverage_ratio"))
        )
        if coverage_ratio is not None:
            metric_catalog[f"report_coverage.{metric_name}"] = coverage_ratio
    return metric_catalog


def _extract_method_metric_value(
    metric_name: str,
    metric_catalog: dict[str, float],
) -> float:
    if metric_name not in metric_catalog:
        raise KeyError(f"matrix metric '{metric_name}' missing from aggregated method metrics")
    return float(metric_catalog[metric_name])


def _evaluate_matrix_metric_constraints(
    metric_catalog: dict[str, float],
    metric_constraints: dict[str, dict[str, float]],
) -> tuple[bool, float, list[str]]:
    if not metric_constraints:
        return True, 0.0, []

    feasible = True
    violation_score = 0.0
    violations: list[str] = []
    for metric_name, bounds in metric_constraints.items():
        value = _extract_method_metric_value(metric_name, metric_catalog)
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


def _rank_matrix_methods(
    rows: list[dict[str, object]],
    selection_mode: str,
) -> list[dict[str, object]]:
    if selection_mode not in {"max", "min"}:
        raise ValueError("matrix_selection_mode must be 'max' or 'min'")

    def row_key(row: dict[str, object]) -> tuple[bool, float, float, float, float, float, int, str]:
        feasible = bool(row.get("matrix_constraint_feasible", True))
        violation_score = float(row.get("matrix_constraint_violation_score", 0.0))
        task_weight_total = float(row["task_weight_total"])
        feasible_task_weight = float(row["feasible_task_weight"])
        selection_value = float(row["matrix_selection_value"])
        selection_key = -selection_value if selection_mode == "max" else selection_value
        mean_rank = float(row["mean_rank"])
        wins = int(row["wins"])
        return (
            not feasible,
            violation_score,
            -task_weight_total,
            -feasible_task_weight,
            selection_key,
            mean_rank,
            -wins,
            str(row["method"]),
        )

    ranked_rows: list[dict[str, object]] = []
    for rank, row in enumerate(sorted(rows, key=row_key), start=1):
        ranked_rows.append({"rank": rank, **{key: value for key, value in row.items() if key != "rank"}})
    return ranked_rows


def _apply_matrix_selection_policy(
    method_leaderboard: list[dict[str, object]],
    task_matrix_rows: list[dict[str, object]],
    method_metric_rows: list[dict[str, object]],
    metric_names: list[str],
    selection_metric: str,
    selection_mode: str,
    metric_constraints: dict[str, dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    metric_rows_by_method = {str(row["method"]): row for row in method_metric_rows}
    annotated_rows: list[dict[str, object]] = []
    for row in method_leaderboard:
        method_name = str(row["method"])
        metric_catalog = _build_method_metric_catalog(
            leaderboard_row=row,
            method_metric_row=metric_rows_by_method.get(method_name),
            metric_names=metric_names,
        )
        selection_value = _extract_method_metric_value(selection_metric, metric_catalog)
        feasible, violation_score, violations = _evaluate_matrix_metric_constraints(
            metric_catalog=metric_catalog,
            metric_constraints=metric_constraints,
        )
        annotated_rows.append(
            {
                **row,
                "matrix_selection_metric": selection_metric,
                "matrix_selection_mode": selection_mode,
                "matrix_selection_value": float(selection_value),
                "matrix_constraint_feasible": bool(feasible),
                "matrix_constraint_violation_score": float(violation_score),
                "matrix_constraint_violation_count": int(len(violations)),
                "matrix_constraint_violations": violations,
            }
        )

    ranked_leaderboard = _rank_matrix_methods(annotated_rows, selection_mode=selection_mode)
    rank_map = {str(row["method"]): int(row["rank"]) for row in ranked_leaderboard}
    constraint_map = {
        str(row["method"]): {
            "matrix_selection_metric": row["matrix_selection_metric"],
            "matrix_selection_mode": row["matrix_selection_mode"],
            "matrix_selection_value": row["matrix_selection_value"],
            "matrix_constraint_feasible": row["matrix_constraint_feasible"],
            "matrix_constraint_violation_score": row["matrix_constraint_violation_score"],
            "matrix_constraint_violation_count": row["matrix_constraint_violation_count"],
        }
        for row in ranked_leaderboard
    }

    ranked_task_matrix_rows = sorted(
        [
            {
                **row,
                "rank": rank_map[str(row["method"])],
                **constraint_map[str(row["method"])],
            }
            for row in task_matrix_rows
        ],
        key=lambda row: (int(row["rank"]), str(row["method"])),
    )
    ranked_method_metric_rows = sorted(
        [
            {
                **row,
                "rank": rank_map[str(row["method"])],
                **constraint_map[str(row["method"])],
            }
            for row in method_metric_rows
        ],
        key=lambda row: (int(row["rank"]), str(row["method"])),
    )
    return ranked_leaderboard, ranked_task_matrix_rows, ranked_method_metric_rows


def _render_markdown_method_metric_report(
    rows: list[dict[str, object]],
    metric_names: list[str],
) -> str:
    headers = ["rank", "method", *metric_names]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, separator_line]
    for row in rows:
        values = [str(row["rank"]), str(row["method"])]
        for metric_name in metric_names:
            mean_value = row.get(_matrix_metric_column_name(metric_name, "weighted_mean"))
            task_weight = float(row.get(_matrix_metric_column_name(metric_name, "task_weight"), 0.0))
            if mean_value is None or task_weight <= 0.0:
                values.append("")
                continue
            values.append(f"{float(mean_value):.6f} (w={task_weight:.2f})")
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _render_markdown_pairwise_comparisons(
    method_names: list[str],
    pairwise_rows: list[dict[str, object]],
) -> str:
    pair_lookup = {
        (str(row["method"]), str(row["opponent"])): row
        for row in pairwise_rows
    }
    headers = ["method", *method_names]
    header_line = "| " + " | ".join(headers) + " |"
    separator_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines = [header_line, separator_line]
    for method_name in method_names:
        values = [method_name]
        for opponent_name in method_names:
            if method_name == opponent_name:
                values.append("-")
                continue
            row = pair_lookup.get((method_name, opponent_name))
            if row is None or int(row["tasks_compared"]) == 0:
                values.append("")
                continue
            values.append(
                f"{int(row['wins'])}-{int(row['losses'])}-{int(row['ties'])} / {float(row['weighted_net']):+.2f}"
            )
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _task_manifest_path(task_dir: Path) -> Path:
    return task_dir / "matrix_task_manifest.json"


def _write_task_manifest(task_dir: Path, generated_spec_path: Path, spec_digest: str) -> None:
    dump_json(
        _task_manifest_path(task_dir),
        {
            "generated_spec_path": str(generated_spec_path.resolve()),
            "spec_digest": spec_digest,
        },
    )


def _load_resumed_task_study(
    task_dir: Path,
    generated_spec_path: Path,
    spec_digest: str,
) -> tuple[StudyArtifacts, dict] | None:
    manifest_path = _task_manifest_path(task_dir)
    summary_path = task_dir / "study_summary.json"
    if not manifest_path.exists() or not summary_path.exists():
        return None

    manifest = _load_json_payload(manifest_path)
    if manifest.get("generated_spec_path") != str(generated_spec_path.resolve()):
        return None
    if manifest.get("spec_digest") != spec_digest:
        return None

    summary = _load_json_payload(summary_path)
    leaderboard_csv_path = task_dir / "leaderboard.csv"
    leaderboard_md_path = task_dir / "leaderboard.md"
    if not leaderboard_csv_path.exists() or not leaderboard_md_path.exists():
        return None

    metrics_csv_path = task_dir / "metrics_report.csv"
    metrics_md_path = task_dir / "metrics_report.md"
    pareto_frontier_path = task_dir / "pareto_frontier.json"
    pareto_frontier_md_path = task_dir / "pareto_frontier.md"
    resolved_config_dir = task_dir / "resolved_configs"
    artifacts = StudyArtifacts(
        root_dir=task_dir,
        summary_path=summary_path,
        leaderboard_csv_path=leaderboard_csv_path,
        leaderboard_md_path=leaderboard_md_path,
        metrics_csv_path=metrics_csv_path if metrics_csv_path.exists() else None,
        metrics_md_path=metrics_md_path if metrics_md_path.exists() else None,
        pareto_frontier_path=pareto_frontier_path if pareto_frontier_path.exists() else None,
        pareto_frontier_md_path=pareto_frontier_md_path if pareto_frontier_md_path.exists() else None,
        resolved_config_dir=resolved_config_dir if resolved_config_dir.exists() else None,
    )
    return artifacts, summary


def run_benchmark_matrix_spec(
    spec_path: str | Path,
    output_dir: str | Path | None = None,
    include_tags: list[str] | None = None,
    exclude_tags: list[str] | None = None,
    resume: bool = True,
) -> tuple[MatrixArtifacts, dict]:
    spec_file = Path(spec_path)
    spec = _load_json_payload(spec_file)

    tasks = spec.get("tasks")
    methods = spec.get("methods")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("matrix spec must contain a non-empty 'tasks' list")
    if not isinstance(methods, list) or not methods:
        raise ValueError("matrix spec must contain a non-empty 'methods' list")

    matrix_root = ensure_dir(output_dir or Path("runs") / f"matrix_{slugify_name(spec_file.stem)}")
    generated_spec_dir = ensure_dir(matrix_root / "generated_specs")
    resolved_task_config_dir = ensure_dir(matrix_root / "resolved_tasks")
    task_output_dir = ensure_dir(matrix_root / "tasks")

    include_tags = _normalize_tag_list(include_tags, field_name="include_tags")
    exclude_tags = _normalize_tag_list(exclude_tags, field_name="exclude_tags")
    global_tags = _normalize_tag_list(spec.get("tags"), field_name="matrix tags")
    matrix_selection_metric = spec.get("matrix_selection_metric")
    if matrix_selection_metric is not None:
        matrix_selection_metric = _normalize_name(matrix_selection_metric, "matrix_selection_metric")
    matrix_selection_mode = str(spec.get("matrix_selection_mode", "max"))
    matrix_metric_constraints = normalize_metric_constraints(spec.get("matrix_metric_constraints"))

    task_results: list[dict[str, object]] = []
    skipped_tasks: list[dict[str, object]] = []
    generated_task_variants = 0
    resumed_task_count = 0
    executed_task_count = 0
    seen_task_variant_names: set[str] = set()

    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("each matrix task must be an object")

        config_ref = task.get("config")
        if not isinstance(config_ref, str):
            raise ValueError("each matrix task needs a string 'config'")

        config_path = (spec_file.parent / config_ref).resolve()
        payload = _load_json_payload(config_path)
        base_name = _normalize_name(task.get("name", payload.get("experiment_name", config_path.stem)), "matrix task name")
        task_tags = _merge_tags(
            global_tags,
            _normalize_tag_list(task.get("tags"), field_name=f"matrix task '{base_name}' tags"),
        )

        task_overrides = task.get("overrides", {})
        if not isinstance(task_overrides, dict):
            raise ValueError(f"matrix task '{base_name}' overrides must be an object")
        task_grid = task.get("grid", {})
        if not isinstance(task_grid, dict):
            raise ValueError(f"matrix task '{base_name}' grid must be an object")

        for variant_overrides, sweep_values, suffix in _expand_grid_variants(task_grid):
            generated_task_variants += 1
            combined_overrides = _merge_override_maps(task_overrides, variant_overrides)
            task_variant_name = _build_task_variant_name(base_name, suffix)
            if task_variant_name in seen_task_variant_names:
                raise ValueError(f"matrix task variants must have unique names, got duplicate '{task_variant_name}'")
            seen_task_variant_names.add(task_variant_name)
            variant_tags = _merge_tags(task_tags, [f"task:{task_variant_name}"])
            task_weight = _normalize_task_weight(
                _resolve_task_setting(task, spec, "task_weight", 1.0),
                field_name=f"matrix task '{task_variant_name}' weight",
            )
            resolved_payload = apply_config_overrides(payload, combined_overrides)
            resolved_task_path = (
                resolved_task_config_dir
                / f"{len(task_results) + len(skipped_tasks) + 1:02d}_{slugify_name(task_variant_name)}.json"
            ).resolve()
            dump_json(resolved_task_path, resolved_payload)

            generated_methods = _build_generated_method_specs(
                methods=methods,
                task_name=task_variant_name,
                task_tags=variant_tags,
                global_tags=[],
            )
            task_mode = str(_resolve_task_setting(task, spec, "mode", "train"))
            generated_task_spec = {
                "base_config": str(resolved_task_path),
                "mode": task_mode,
                "selection_metric": _resolve_task_setting(task, spec, "selection_metric", "mean_terminal_score"),
                "selection_mode": _resolve_task_setting(task, spec, "selection_mode", "max"),
                "metric_constraints": _resolve_task_setting(task, spec, "metric_constraints", {}),
                "report_metrics": _resolve_task_setting(task, spec, "report_metrics", []),
                "pareto_metrics": _resolve_task_setting(task, spec, "pareto_metrics", {}),
                "benchmark_seeds": _resolve_task_setting(task, spec, "benchmark_seeds", None),
                "walk_forward_folds": _resolve_task_setting(task, spec, "walk_forward_folds", None),
                "train_ratio_start": float(_resolve_task_setting(task, spec, "train_ratio_start", 0.5)),
                "experiments": generated_methods,
            }
            generated_task_spec_digest = _compute_payload_digest(generated_task_spec)
            generated_spec_path = generated_spec_dir / f"{len(task_results) + len(skipped_tasks) + 1:02d}_{slugify_name(task_variant_name)}.json"
            dump_json(generated_spec_path, generated_task_spec)

            task_dir = ensure_dir(task_output_dir / f"{len(task_results) + len(skipped_tasks) + 1:02d}_{slugify_name(task_variant_name)}")
            resumed_payload = None
            if resume:
                resumed_payload = _load_resumed_task_study(
                    task_dir=task_dir,
                    generated_spec_path=generated_spec_path,
                    spec_digest=generated_task_spec_digest,
                )
            if resumed_payload is not None:
                study_artifacts, study_summary = resumed_payload
                resumed_task_count += 1
                task_results.append(
                    {
                        "task": task_variant_name,
                        "config_path": str(config_path),
                        "resolved_task_path": str(resolved_task_path),
                        "generated_spec_path": str(generated_spec_path),
                        "tags": variant_tags,
                        "task_weight": float(task_weight),
                        "task_status": "resumed",
                        "sweep_values": sweep_values,
                        "artifacts": study_artifacts,
                        "summary": study_summary,
                    }
                )
                continue
            try:
                study_artifacts, study_summary = run_study_spec(
                    generated_spec_path,
                    output_dir=task_dir,
                    include_tags=include_tags,
                    exclude_tags=exclude_tags,
                )
            except ValueError as exc:
                if "did not produce any experiments after tag filtering" not in str(exc):
                    raise
                skipped_tasks.append(
                    {
                        "task": task_variant_name,
                        "config_path": str(config_path),
                        "resolved_task_path": str(resolved_task_path),
                        "generated_spec_path": str(generated_spec_path),
                        "tags": variant_tags,
                        "task_weight": float(task_weight),
                        "reason": "filtered_out",
                    }
                )
                continue

            _write_task_manifest(
                task_dir=task_dir,
                generated_spec_path=generated_spec_path,
                spec_digest=generated_task_spec_digest,
            )
            executed_task_count += 1
            task_results.append(
                {
                    "task": task_variant_name,
                    "config_path": str(config_path),
                    "resolved_task_path": str(resolved_task_path),
                    "generated_spec_path": str(generated_spec_path),
                    "tags": variant_tags,
                    "task_weight": float(task_weight),
                    "task_status": "executed",
                    "sweep_values": sweep_values,
                    "artifacts": study_artifacts,
                    "summary": study_summary,
                }
            )

    if not task_results:
        raise ValueError("matrix spec did not produce any runnable task variants after tag filtering")

    method_leaderboard, task_matrix_rows = _build_method_leaderboard(task_results)
    matrix_report_metrics, method_metric_rows = _build_method_metric_report(task_results, method_leaderboard)
    if matrix_selection_metric is not None or matrix_metric_constraints:
        resolved_matrix_selection_metric = matrix_selection_metric or "matrix.mean_normalized_rank_score"
        method_leaderboard, task_matrix_rows, method_metric_rows = _apply_matrix_selection_policy(
            method_leaderboard=method_leaderboard,
            task_matrix_rows=task_matrix_rows,
            method_metric_rows=method_metric_rows,
            metric_names=matrix_report_metrics,
            selection_metric=resolved_matrix_selection_metric,
            selection_mode=matrix_selection_mode,
            metric_constraints=matrix_metric_constraints,
        )
    pairwise_method_names, pairwise_rows = _build_pairwise_comparisons(task_results)
    task_names = [str(task_result["task"]) for task_result in task_results]
    task_weight_map = {
        str(task_result["task"]): float(task_result["task_weight"])
        for task_result in task_results
    }

    leaderboard_csv_path = matrix_root / "matrix_leaderboard.csv"
    leaderboard_md_path = matrix_root / "matrix_leaderboard.md"
    task_matrix_csv_path = matrix_root / "task_matrix.csv"
    task_matrix_md_path = matrix_root / "task_matrix.md"
    pairwise_csv_path = matrix_root / "pairwise_wins.csv"
    pairwise_md_path = matrix_root / "pairwise_wins.md"
    method_metrics_csv_path = matrix_root / "method_metrics.csv" if matrix_report_metrics else None
    method_metrics_md_path = matrix_root / "method_metrics.md" if matrix_report_metrics else None
    summary_path = matrix_root / "matrix_summary.json"

    dump_records_csv(leaderboard_csv_path, method_leaderboard)
    leaderboard_md_path.write_text(_render_markdown_matrix_leaderboard(method_leaderboard), encoding="utf-8")
    dump_records_csv(task_matrix_csv_path, task_matrix_rows)
    task_matrix_md_path.write_text(
        _render_markdown_task_matrix(task_matrix_rows, task_names=task_names, task_weights=task_weight_map),
        encoding="utf-8",
    )
    dump_records_csv(pairwise_csv_path, pairwise_rows)
    pairwise_md_path.write_text(
        _render_markdown_pairwise_comparisons(pairwise_method_names, pairwise_rows),
        encoding="utf-8",
    )
    if method_metrics_csv_path is not None and method_metrics_md_path is not None:
        dump_records_csv(method_metrics_csv_path, method_metric_rows)
        method_metrics_md_path.write_text(
            _render_markdown_method_metric_report(method_metric_rows, metric_names=matrix_report_metrics),
            encoding="utf-8",
        )

    summary = {
        "spec_path": str(spec_file),
        "include_tags": include_tags,
        "exclude_tags": exclude_tags,
        "generated_task_variants": int(generated_task_variants),
        "task_count": int(len(task_results)),
        "total_task_weight": float(sum(task_weight_map.values())),
        "skipped_task_count": int(len(skipped_tasks)),
        "resumed_task_count": int(resumed_task_count),
        "executed_task_count": int(executed_task_count),
        "top_method": method_leaderboard[0]["method"],
        "matrix_selection_metric": matrix_selection_metric or ("matrix.mean_normalized_rank_score" if matrix_metric_constraints else None),
        "matrix_selection_mode": matrix_selection_mode if matrix_selection_metric is not None or matrix_metric_constraints else None,
        "matrix_metric_constraints": matrix_metric_constraints,
        "leaderboard": method_leaderboard,
        "task_matrix": task_matrix_rows,
        "matrix_report_metrics": matrix_report_metrics,
        "method_metric_report": method_metric_rows,
        "pairwise_comparisons": pairwise_rows,
        "tasks": [
            _build_task_summary_record(
                task_name=str(task_result["task"]),
                task_tags=list(task_result["tags"]),
                task_weight=float(task_result["task_weight"]),
                task_status=str(task_result["task_status"]),
                study_artifacts=task_result["artifacts"],
                study_summary=task_result["summary"],
            )
            for task_result in task_results
        ],
        "skipped_tasks": skipped_tasks,
    }
    dump_json(summary_path, summary)

    artifacts = MatrixArtifacts(
        root_dir=matrix_root,
        summary_path=summary_path,
        leaderboard_csv_path=leaderboard_csv_path,
        leaderboard_md_path=leaderboard_md_path,
        task_matrix_csv_path=task_matrix_csv_path,
        task_matrix_md_path=task_matrix_md_path,
        pairwise_csv_path=pairwise_csv_path,
        pairwise_md_path=pairwise_md_path,
        generated_spec_dir=generated_spec_dir,
        resolved_task_config_dir=resolved_task_config_dir,
        task_output_dir=task_output_dir,
        method_metrics_csv_path=method_metrics_csv_path,
        method_metrics_md_path=method_metrics_md_path,
    )
    return artifacts, summary
