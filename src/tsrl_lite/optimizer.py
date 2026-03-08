from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from math import prod
from pathlib import Path
from time import sleep

from tsrl_lite.matrix import run_benchmark_matrix_spec
from tsrl_lite.study import (
    _evaluate_study_item,
    _finalize_study_outputs,
    apply_config_overrides,
    normalize_metric_constraints,
    normalize_metric_directions,
    normalize_metric_name_list,
    prepare_metric_report_names,
    slugify_name,
)
from tsrl_lite.utils import dump_json, dump_records_csv, ensure_dir


@dataclass(slots=True)
class OvernightArtifacts:
    root_dir: Path
    state_path: Path
    heartbeat_path: Path
    stop_path: Path
    spec_copy_path: Path
    progress_csv_path: Path
    best_config_path: Path
    best_candidate_path: Path


@dataclass(slots=True)
class OvernightWatchdogArtifacts:
    root_dir: Path
    state_path: Path
    heartbeat_path: Path
    stop_path: Path
    optimizer_state_path: Path
    optimizer_heartbeat_path: Path


def _extract_numeric(value: object) -> float | None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    return float(value)


def _clear_active_generation_state(state: dict[str, object]) -> None:
    state["active_generation"] = None
    state["active_generation_dir"] = None
    state["active_candidates_completed"] = 0
    state["active_candidates_total"] = 0
    state["active_population"] = []
    state["active_generation_rows"] = []
    state["active_generation_config_paths"] = []


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_deadline(deadline: str | None) -> datetime | None:
    if deadline is None:
        return None
    parsed = datetime.fromisoformat(deadline)
    if parsed.tzinfo is None:
        raise ValueError("deadline must include an explicit timezone offset")
    return parsed.astimezone(timezone.utc)


def _candidate_key(candidate: dict[str, object]) -> str:
    return json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sample_candidate(search_space: dict[str, list[object]], rng: random.Random) -> dict[str, object]:
    candidate: dict[str, object] = {}
    for key, values in search_space.items():
        candidate[key] = copy.deepcopy(rng.choice(values))
    return candidate


def _mutate_candidate(
    parent: dict[str, object],
    search_space: dict[str, list[object]],
    mutation_rate: float,
    rng: random.Random,
) -> dict[str, object]:
    child = copy.deepcopy(parent)
    mutable_keys = [key for key, values in search_space.items() if len(values) > 1]
    mutated = False

    for key in mutable_keys:
        if rng.random() >= mutation_rate:
            continue
        current_key = json.dumps(child[key], sort_keys=True, ensure_ascii=True)
        alternatives = [
            value
            for value in search_space[key]
            if json.dumps(value, sort_keys=True, ensure_ascii=True) != current_key
        ]
        if not alternatives:
            continue
        child[key] = copy.deepcopy(rng.choice(alternatives))
        mutated = True

    if mutated or not mutable_keys:
        return child

    key = rng.choice(mutable_keys)
    current_key = json.dumps(child[key], sort_keys=True, ensure_ascii=True)
    alternatives = [
        value for value in search_space[key] if json.dumps(value, sort_keys=True, ensure_ascii=True) != current_key
    ]
    if alternatives:
        child[key] = copy.deepcopy(rng.choice(alternatives))
    return child


def _rank_candidates(
    candidates: list[dict[str, object]],
    selection_mode: str,
) -> list[dict[str, object]]:
    def candidate_key(item: dict[str, object]) -> tuple[bool, float, float]:
        feasible = bool(item.get("constraint_feasible", True))
        violation_score = float(item.get("constraint_violation_score", 0.0))
        selection_value = float(item["selection_value"])
        selection_key = -selection_value if selection_mode == "max" else selection_value
        return (not feasible, violation_score, selection_key)

    return sorted(candidates, key=candidate_key)


def _load_state(state_path: Path) -> dict[str, object] | None:
    if not state_path.exists():
        return None
    return _load_json(state_path)


def _write_heartbeat(heartbeat_path: Path, payload: dict[str, object]) -> None:
    dump_json(heartbeat_path, payload)


def _write_progress_csv(progress_csv_path: Path, evaluated_candidates: list[dict[str, object]]) -> None:
    rows: list[dict[str, object]] = []
    for entry in evaluated_candidates:
        row = {
            "generation": int(entry["generation"]),
            "candidate_index": int(entry["candidate_index"]),
            "selection_value": float(entry["selection_value"]),
            "experiment": str(entry["experiment"]),
            "summary_path": str(entry["summary_path"]),
            "rank": int(entry["rank"]),
            "constraint_feasible": bool(entry.get("constraint_feasible", True)),
            "constraint_violation_score": float(entry.get("constraint_violation_score", 0.0)),
            "constraint_violation_count": int(entry.get("constraint_violation_count", 0)),
        }
        for key, value in dict(entry["params"]).items():
            row[f"param_{slugify_name(key.replace('.', '_'))}"] = value
        rows.append(row)
    dump_records_csv(progress_csv_path, rows)


def _write_best_snapshots(
    best_config_path: Path,
    best_candidate_path: Path,
    base_payload: dict[str, object],
    fixed_overrides: dict[str, object],
    best_candidate: dict[str, object] | None,
) -> None:
    if best_candidate is None:
        return
    best_payload = apply_config_overrides(base_payload, fixed_overrides)
    best_payload = apply_config_overrides(best_payload, dict(best_candidate["params"]))
    best_payload["experiment_name"] = str(best_candidate["experiment"])
    dump_json(best_config_path, best_payload)
    dump_json(best_candidate_path, best_candidate)


def _write_best_matrix_snapshots(
    best_config_path: Path,
    best_candidate_path: Path,
    base_matrix_payload: dict[str, object],
    target_method: str,
    fixed_overrides: dict[str, object],
    best_candidate: dict[str, object] | None,
) -> None:
    if best_candidate is None:
        return
    best_payload = _build_matrix_candidate_payload(
        base_matrix_payload=base_matrix_payload,
        target_method=target_method,
        fixed_overrides=fixed_overrides,
        candidate_params=dict(best_candidate["params"]),
    )
    dump_json(best_config_path, best_payload)
    dump_json(best_candidate_path, best_candidate)


def _build_matrix_candidate_payload(
    base_matrix_payload: dict[str, object],
    target_method: str,
    fixed_overrides: dict[str, object],
    candidate_params: dict[str, object],
) -> dict[str, object]:
    payload = copy.deepcopy(base_matrix_payload)
    methods = payload.get("methods")
    if not isinstance(methods, list):
        raise ValueError("matrix optimizer base spec must contain a 'methods' list")

    matched_method = None
    for method in methods:
        if isinstance(method, dict) and method.get("name") == target_method:
            matched_method = method
            break
    if matched_method is None:
        raise ValueError(f"target_method '{target_method}' not found in matrix spec methods")

    existing_overrides = matched_method.get("overrides", {})
    if existing_overrides is None:
        existing_overrides = {}
    if not isinstance(existing_overrides, dict):
        raise ValueError(f"matrix method '{target_method}' overrides must be an object")
    resolved_overrides = apply_config_overrides(existing_overrides, fixed_overrides)
    resolved_overrides = apply_config_overrides(resolved_overrides, candidate_params)
    matched_method["overrides"] = resolved_overrides
    return payload


def _resolve_matrix_task_config_paths(
    matrix_payload: dict[str, object],
    matrix_spec_path: Path,
) -> dict[str, object]:
    resolved_payload = copy.deepcopy(matrix_payload)
    tasks = resolved_payload.get("tasks")
    if not isinstance(tasks, list):
        raise ValueError("matrix optimizer base spec must contain a 'tasks' list")
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("matrix optimizer tasks must be objects")
        config_ref = task.get("config")
        if not isinstance(config_ref, str):
            raise ValueError("matrix optimizer tasks must define a string 'config'")
        task["config"] = str((matrix_spec_path.parent / config_ref).resolve())
    return resolved_payload


def _build_matrix_candidate_metric_catalog(
    matrix_row: dict[str, object],
    method_metric_row: dict[str, object],
    matrix_report_metrics: list[str],
) -> dict[str, float]:
    metric_catalog: dict[str, float] = {}
    for key, value in matrix_row.items():
        numeric_value = _extract_numeric(value)
        if numeric_value is None:
            continue
        metric_catalog[f"matrix.{key}"] = numeric_value

    for metric_name in matrix_report_metrics:
        slug = slugify_name(metric_name)
        weighted_mean = _extract_numeric(method_metric_row.get(f"metric_{slug}_weighted_mean"))
        if weighted_mean is not None:
            metric_catalog[metric_name] = weighted_mean
            metric_catalog[f"report.{metric_name}"] = weighted_mean
        task_weight = _extract_numeric(method_metric_row.get(f"metric_{slug}_task_weight"))
        if task_weight is not None:
            metric_catalog[f"report_task_weight.{metric_name}"] = task_weight
        coverage_ratio = _extract_numeric(method_metric_row.get(f"metric_{slug}_weighted_coverage_ratio"))
        if coverage_ratio is not None:
            metric_catalog[f"report_coverage.{metric_name}"] = coverage_ratio
    return metric_catalog


def _extract_optimizer_metric_value(
    metric_name: str,
    metric_catalog: dict[str, float],
) -> float:
    if metric_name not in metric_catalog:
        raise KeyError(f"optimizer metric '{metric_name}' missing from available candidate metrics")
    return float(metric_catalog[metric_name])


def _evaluate_optimizer_metric_constraints(
    metric_catalog: dict[str, float],
    metric_constraints: dict[str, dict[str, float]],
) -> tuple[bool, float, list[str]]:
    if not metric_constraints:
        return True, 0.0, []

    feasible = True
    violation_score = 0.0
    violations: list[str] = []
    for metric_name, bounds in metric_constraints.items():
        value = _extract_optimizer_metric_value(metric_name, metric_catalog)
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


def _evaluate_matrix_candidate(
    *,
    spec_path: Path,
    output_dir: Path,
    experiment_name: str,
    target_method: str,
    selection_metric: str,
    metric_constraints: dict[str, dict[str, float]],
    metric_report_names: list[str],
    row_fields: dict[str, object],
) -> dict[str, object]:
    artifacts, summary = run_benchmark_matrix_spec(
        spec_path,
        output_dir=output_dir,
        resume=False,
    )
    leaderboard = summary.get("leaderboard", [])
    if not isinstance(leaderboard, list):
        raise KeyError("matrix summary is missing 'leaderboard'")
    matrix_row = next((row for row in leaderboard if row.get("method") == target_method), None)
    if matrix_row is None:
        raise KeyError(f"target method '{target_method}' missing from matrix leaderboard")

    method_metric_rows = summary.get("method_metric_report", [])
    if not isinstance(method_metric_rows, list):
        method_metric_rows = []
    method_metric_row = next((row for row in method_metric_rows if row.get("method") == target_method), {})
    if not isinstance(method_metric_row, dict):
        method_metric_row = {}
    matrix_report_metrics = summary.get("matrix_report_metrics", [])
    if not isinstance(matrix_report_metrics, list):
        matrix_report_metrics = []
    metric_catalog = _build_matrix_candidate_metric_catalog(
        matrix_row=matrix_row,
        method_metric_row=method_metric_row,
        matrix_report_metrics=[metric for metric in matrix_report_metrics if isinstance(metric, str)],
    )

    selection_value = _extract_optimizer_metric_value(selection_metric, metric_catalog)
    base_feasible = bool(matrix_row.get("matrix_constraint_feasible", True))
    base_violation_score = float(matrix_row.get("matrix_constraint_violation_score", 0.0))
    base_violations = list(matrix_row.get("matrix_constraint_violations", []))
    extra_feasible, extra_violation_score, extra_violations = _evaluate_optimizer_metric_constraints(
        metric_catalog=metric_catalog,
        metric_constraints=metric_constraints,
    )
    requested_metric_values = {
        metric_name: _extract_optimizer_metric_value(metric_name, metric_catalog)
        for metric_name in metric_report_names
    }

    row = {
        "experiment": experiment_name,
        "config_path": str(spec_path),
        "mode": "matrix",
        "output_dir": str(output_dir),
        "summary_path": str(artifacts.summary_path),
        "env": "matrix",
        "encoder": target_method,
        "agent": "matrix",
        "selection_metric": selection_metric,
        "selection_value": float(selection_value),
        "constraint_feasible": bool(base_feasible and extra_feasible),
        "constraint_violation_score": float(base_violation_score + extra_violation_score),
        "constraint_violation_count": int(len(base_violations) + len(extra_violations)),
        "constraint_violations": [*base_violations, *extra_violations],
        "matrix_target_method": target_method,
    }
    for metric_name, metric_value in requested_metric_values.items():
        row[f"report_{slugify_name(metric_name)}"] = float(metric_value)
    for key, value in matrix_row.items():
        numeric_value = _extract_numeric(value)
        if numeric_value is None:
            continue
        row[f"matrix_{key}"] = numeric_value
    for key, value in method_metric_row.items():
        numeric_value = _extract_numeric(value)
        if numeric_value is None:
            continue
        row[f"method_{key}"] = numeric_value
    for key, value in row_fields.items():
        row[key] = value
    return row


def run_overnight_watchdog(
    spec_path: str | Path,
    output_dir: str | Path | None = None,
    deadline: str | None = None,
    max_generations: int | None = None,
    population_size: int | None = None,
    elite_size: int | None = None,
    mutation_rate: float | None = None,
    stop_file: str | None = None,
    max_restarts: int | None = None,
    restart_delay_seconds: float = 5.0,
) -> tuple[OvernightWatchdogArtifacts, dict]:
    if restart_delay_seconds < 0.0:
        raise ValueError("restart_delay_seconds must be >= 0")

    spec_file = Path(spec_path)
    spec = _load_json(spec_file)
    optimizer_name = str(spec.get("name", spec_file.stem))
    optimizer_root = ensure_dir(output_dir or Path("runs") / slugify_name(optimizer_name))
    stop_path = optimizer_root / str(stop_file or spec.get("stop_file", "STOP"))
    watchdog_state_path = optimizer_root / "watchdog_state.json"
    watchdog_heartbeat_path = optimizer_root / "watchdog_heartbeat.json"
    optimizer_state_path = optimizer_root / "state.json"
    optimizer_heartbeat_path = optimizer_root / "heartbeat.json"
    resolved_deadline_raw = deadline or spec.get("deadline")
    resolved_deadline = _parse_deadline(str(resolved_deadline_raw)) if resolved_deadline_raw else None

    state = _load_state(watchdog_state_path) or {
        "optimizer_name": optimizer_name,
        "spec_path": str(spec_file),
        "started_at": _utc_now().isoformat(),
        "launch_count": 0,
        "restart_count": 0,
        "last_error": None,
        "last_optimizer_status": None,
        "status": "running",
    }

    artifacts = OvernightWatchdogArtifacts(
        root_dir=optimizer_root,
        state_path=watchdog_state_path,
        heartbeat_path=watchdog_heartbeat_path,
        stop_path=stop_path,
        optimizer_state_path=optimizer_state_path,
        optimizer_heartbeat_path=optimizer_heartbeat_path,
    )

    while True:
        now = _utc_now()
        state["updated_at"] = now.isoformat()
        heartbeat = {
            "optimizer_name": optimizer_name,
            "status": state["status"],
            "deadline": resolved_deadline.astimezone().isoformat() if resolved_deadline else None,
            "updated_at": state["updated_at"],
            "launch_count": int(state["launch_count"]),
            "restart_count": int(state["restart_count"]),
            "last_optimizer_status": state.get("last_optimizer_status"),
            "last_error": state.get("last_error"),
            "stop_file": str(stop_path),
        }
        dump_json(watchdog_state_path, state)
        _write_heartbeat(watchdog_heartbeat_path, heartbeat)

        if resolved_deadline and now >= resolved_deadline:
            state["status"] = "deadline_reached"
        if stop_path.exists():
            state["status"] = "stop_file_detected"
        if max_restarts is not None and int(state["restart_count"]) >= max_restarts:
            state["status"] = "max_restarts_exceeded"
        if state["status"] in {"deadline_reached", "stop_file_detected", "max_restarts_exceeded"}:
            dump_json(watchdog_state_path, state)
            _write_heartbeat(
                watchdog_heartbeat_path,
                {
                    **heartbeat,
                    "status": state["status"],
                    "updated_at": _utc_now().isoformat(),
                },
            )
            return artifacts, state

        state["status"] = "launching_optimizer"
        state["launch_count"] = int(state["launch_count"]) + 1
        state["updated_at"] = _utc_now().isoformat()
        dump_json(watchdog_state_path, state)
        _write_heartbeat(
            watchdog_heartbeat_path,
            {
                **heartbeat,
                "status": state["status"],
                "updated_at": state["updated_at"],
                "launch_count": int(state["launch_count"]),
            },
        )

        try:
            _, optimizer_state = run_overnight_optimizer(
                spec_path=spec_path,
                output_dir=optimizer_root,
                deadline=deadline,
                max_generations=max_generations,
                population_size=population_size,
                elite_size=elite_size,
                mutation_rate=mutation_rate,
                stop_file=stop_file,
            )
        except Exception as exc:
            state["restart_count"] = int(state["restart_count"]) + 1
            state["status"] = "optimizer_crashed"
            state["last_error"] = repr(exc)
            state["last_crash_at"] = _utc_now().isoformat()
            state["updated_at"] = state["last_crash_at"]
            dump_json(watchdog_state_path, state)
            _write_heartbeat(
                watchdog_heartbeat_path,
                {
                    **heartbeat,
                    "status": state["status"],
                    "updated_at": state["updated_at"],
                    "launch_count": int(state["launch_count"]),
                    "restart_count": int(state["restart_count"]),
                    "last_error": state["last_error"],
                },
            )
            if max_restarts is not None and int(state["restart_count"]) >= max_restarts:
                state["status"] = "max_restarts_exceeded"
                continue
            if resolved_deadline and _utc_now() >= resolved_deadline:
                state["status"] = "deadline_reached"
                continue
            if stop_path.exists():
                state["status"] = "stop_file_detected"
                continue
            sleep(restart_delay_seconds)
            state["status"] = "running"
            continue

        state["last_optimizer_status"] = optimizer_state["status"]
        state["best_candidate"] = optimizer_state.get("best_candidate")
        state["last_error"] = None
        state["status"] = str(optimizer_state["status"])
        state["updated_at"] = _utc_now().isoformat()
        dump_json(watchdog_state_path, state)
        _write_heartbeat(
            watchdog_heartbeat_path,
            {
                **heartbeat,
                "status": state["status"],
                "updated_at": state["updated_at"],
                "launch_count": int(state["launch_count"]),
                "restart_count": int(state["restart_count"]),
                "last_optimizer_status": state["last_optimizer_status"],
                "last_error": None,
            },
        )
        return artifacts, state


def run_overnight_optimizer(
    spec_path: str | Path,
    output_dir: str | Path | None = None,
    deadline: str | None = None,
    max_generations: int | None = None,
    population_size: int | None = None,
    elite_size: int | None = None,
    mutation_rate: float | None = None,
    stop_file: str | None = None,
) -> tuple[OvernightArtifacts, dict]:
    spec_file = Path(spec_path)
    spec = _load_json(spec_file)

    base_config_ref = spec.get("base_config")
    base_matrix_ref = spec.get("base_matrix_spec")
    optimizer_kind = "matrix_method" if base_matrix_ref is not None else "config"
    if base_config_ref is not None and base_matrix_ref is not None:
        raise ValueError("optimizer spec must not define both 'base_config' and 'base_matrix_spec'")
    if optimizer_kind == "config":
        if not isinstance(base_config_ref, str):
            raise ValueError("optimizer spec must contain a string 'base_config'")
    else:
        if not isinstance(base_matrix_ref, str):
            raise ValueError("matrix optimizer spec must contain a string 'base_matrix_spec'")
        target_method = spec.get("target_method")
        if not isinstance(target_method, str) or not target_method.strip():
            raise ValueError("matrix optimizer spec must contain a non-empty string 'target_method'")
        target_method = target_method.strip()
    search_space_raw = spec.get("search_space")
    if not isinstance(search_space_raw, dict) or not search_space_raw:
        raise ValueError("optimizer spec must contain a non-empty 'search_space' object")

    search_space: dict[str, list[object]] = {}
    for key, values in search_space_raw.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"search_space field '{key}' must be a non-empty list")
        search_space[str(key)] = values

    resolved_deadline_raw = deadline or spec.get("deadline")
    resolved_deadline = _parse_deadline(str(resolved_deadline_raw)) if resolved_deadline_raw else None
    resolved_population_size = int(population_size or spec.get("population_size", 4))
    resolved_elite_size = int(elite_size or spec.get("elite_size", 2))
    resolved_mutation_rate = float(mutation_rate if mutation_rate is not None else spec.get("mutation_rate", 0.35))
    resolved_max_generations = (
        int(max_generations if max_generations is not None else spec["max_generations"])
        if spec.get("max_generations") is not None or max_generations is not None
        else None
    )
    if resolved_population_size < 1:
        raise ValueError("population_size must be >= 1")
    if resolved_elite_size < 1:
        raise ValueError("elite_size must be >= 1")
    if resolved_mutation_rate <= 0.0 or resolved_mutation_rate > 1.0:
        raise ValueError("mutation_rate must be in (0, 1]")

    optimizer_name = str(spec.get("name", spec_file.stem))
    optimizer_root = ensure_dir(output_dir or Path("runs") / slugify_name(optimizer_name))
    state_path = optimizer_root / "state.json"
    heartbeat_path = optimizer_root / "heartbeat.json"
    stop_path = optimizer_root / str(stop_file or spec.get("stop_file", "STOP"))
    spec_copy_path = optimizer_root / "optimizer_spec.json"
    progress_csv_path = optimizer_root / "progress.csv"
    best_config_path = optimizer_root / "best_config.json"
    best_candidate_path = optimizer_root / "best_candidate.json"
    base_payload_path = (
        (spec_file.parent / str(base_config_ref)).resolve()
        if optimizer_kind == "config"
        else (spec_file.parent / str(base_matrix_ref)).resolve()
    )
    base_payload = _load_json(base_payload_path)
    if optimizer_kind == "matrix_method":
        base_payload = _resolve_matrix_task_config_paths(base_payload, matrix_spec_path=base_payload_path)
    fixed_overrides = spec.get("fixed_overrides", {})
    if not isinstance(fixed_overrides, dict):
        raise ValueError("fixed_overrides must be an object")

    state = _load_state(state_path) or {
        "optimizer_name": optimizer_name,
        "optimizer_kind": optimizer_kind,
        "spec_path": str(spec_file),
        "base_config_path": str(base_payload_path),
        "started_at": _utc_now().isoformat(),
        "generation": 0,
        "completed_generations": [],
        "evaluated_candidates": [],
        "best_candidate": None,
        "status": "running",
        "active_generation": None,
        "active_generation_dir": None,
        "active_candidates_completed": 0,
        "active_candidates_total": 0,
        "active_population": [],
        "active_generation_rows": [],
        "active_generation_config_paths": [],
    }
    state["optimizer_kind"] = optimizer_kind
    if optimizer_kind == "matrix_method":
        state["target_method"] = target_method
    state.setdefault("active_generation", None)
    state.setdefault("active_generation_dir", None)
    state.setdefault("active_candidates_completed", 0)
    state.setdefault("active_candidates_total", 0)
    state.setdefault("active_population", [])
    state.setdefault("active_generation_rows", [])
    state.setdefault("active_generation_config_paths", [])
    dump_json(spec_copy_path, spec)

    seed = int(spec.get("seed", 7))
    rng = random.Random(seed + int(state["generation"]))
    total_search_space = int(prod(len(values) for values in search_space.values()))
    mode = "matrix" if optimizer_kind == "matrix_method" else str(spec.get("mode", "train"))
    if optimizer_kind == "config":
        selection_metric = str(spec.get("selection_metric", "mean_terminal_score"))
        selection_mode = str(spec.get("selection_mode", "max"))
    else:
        raw_matrix_selection_metric = spec.get("selection_metric")
        if raw_matrix_selection_metric is None:
            raw_matrix_selection_metric = base_payload.get("matrix_selection_metric")
        if raw_matrix_selection_metric is None:
            raw_matrix_selection_metric = "matrix.mean_normalized_rank_score"
        selection_metric = str(raw_matrix_selection_metric)

        raw_matrix_selection_mode = spec.get("selection_mode")
        if raw_matrix_selection_mode is None:
            raw_matrix_selection_mode = base_payload.get("matrix_selection_mode")
        if raw_matrix_selection_mode is None:
            raw_matrix_selection_mode = "max"
        selection_mode = str(raw_matrix_selection_mode)
    metric_constraints = normalize_metric_constraints(spec.get("metric_constraints"))
    pareto_metrics = normalize_metric_directions(spec.get("pareto_metrics"), field_name="pareto_metrics")
    report_metrics = normalize_metric_name_list(spec.get("report_metrics"), field_name="report_metrics")
    metric_report_names = prepare_metric_report_names(
        selection_metric=selection_metric,
        metric_constraints=metric_constraints,
        report_metrics=report_metrics,
        pareto_metrics=pareto_metrics,
    )

    artifacts = OvernightArtifacts(
        root_dir=optimizer_root,
        state_path=state_path,
        heartbeat_path=heartbeat_path,
        stop_path=stop_path,
        spec_copy_path=spec_copy_path,
        progress_csv_path=progress_csv_path,
        best_config_path=best_config_path,
        best_candidate_path=best_candidate_path,
    )

    while True:
        now = _utc_now()
        ranked_history = _rank_candidates(
            list(state["evaluated_candidates"]),
            selection_mode=selection_mode,
        )
        best_candidate = ranked_history[0] if ranked_history else None
        state["best_candidate"] = best_candidate
        state["updated_at"] = now.isoformat()
        _write_progress_csv(progress_csv_path, list(state["evaluated_candidates"]))
        if optimizer_kind == "config":
            _write_best_snapshots(
                best_config_path=best_config_path,
                best_candidate_path=best_candidate_path,
                base_payload=base_payload,
                fixed_overrides=fixed_overrides,
                best_candidate=best_candidate,
            )
        else:
            _write_best_matrix_snapshots(
                best_config_path=best_config_path,
                best_candidate_path=best_candidate_path,
                base_matrix_payload=base_payload,
                target_method=target_method,
                fixed_overrides=fixed_overrides,
                best_candidate=best_candidate,
            )

        heartbeat = {
            "optimizer_name": optimizer_name,
            "optimizer_kind": optimizer_kind,
            "status": state["status"],
            "generation": int(state["generation"]),
            "active_generation": state.get("active_generation"),
            "active_candidates_completed": int(state.get("active_candidates_completed", 0)),
            "active_candidates_total": int(state.get("active_candidates_total", 0)),
            "deadline": resolved_deadline.astimezone().isoformat() if resolved_deadline else None,
            "updated_at": now.isoformat(),
            "best_candidate": best_candidate,
            "stop_file": str(stop_path),
        }
        _write_heartbeat(heartbeat_path, heartbeat)
        dump_json(state_path, state)

        if resolved_deadline and now >= resolved_deadline:
            state["status"] = "deadline_reached"
            break
        if stop_path.exists():
            state["status"] = "stop_file_detected"
            break
        if resolved_max_generations is not None and int(state["generation"]) >= resolved_max_generations:
            state["status"] = "max_generations_reached"
            break
        if len(state["evaluated_candidates"]) >= total_search_space:
            state["status"] = "search_space_exhausted"
            break

        generation_interrupted = False
        active_population = state.get("active_population", [])
        resume_generation = (
            state.get("active_generation") is not None
            and isinstance(active_population, list)
            and bool(active_population)
            and int(state.get("active_candidates_completed", 0)) < len(active_population)
        )

        if resume_generation:
            generation_index = int(state["active_generation"])
            generation_dir_ref = state.get("active_generation_dir")
            generation_dir = ensure_dir(
                Path(generation_dir_ref)
                if isinstance(generation_dir_ref, str) and generation_dir_ref
                else optimizer_root / f"generation_{generation_index:04d}"
            )
            config_dir = ensure_dir(generation_dir / "configs")
            generation_candidates = [copy.deepcopy(candidate) for candidate in active_population]
            generation_rows = [copy.deepcopy(row) for row in state.get("active_generation_rows", [])]
            generation_config_paths = [
                Path(path) for path in state.get("active_generation_config_paths", [])
            ]
            start_candidate_index = int(state.get("active_candidates_completed", 0)) + 1
            state["status"] = "running_generation"
            state["active_generation"] = generation_index
            state["active_generation_dir"] = str(generation_dir)
            state["active_candidates_total"] = len(generation_candidates)
        else:
            generation_index = int(state["generation"]) + 1
            generation_dir = ensure_dir(optimizer_root / f"generation_{generation_index:04d}")
            config_dir = ensure_dir(generation_dir / "configs")

            seen_keys = {entry["candidate_key"] for entry in state["evaluated_candidates"]}
            elites = [entry["params"] for entry in ranked_history[:resolved_elite_size]]
            generation_candidates: list[dict[str, object]] = []
            generation_keys: set[str] = set()
            max_attempts = max(64, resolved_population_size * 32)

            attempts = 0
            while len(generation_candidates) < resolved_population_size and attempts < max_attempts:
                attempts += 1
                if elites and rng.random() < 0.7:
                    parent = copy.deepcopy(rng.choice(elites))
                    candidate = _mutate_candidate(parent, search_space, resolved_mutation_rate, rng)
                else:
                    candidate = _sample_candidate(search_space, rng)
                candidate_key = _candidate_key(candidate)
                if candidate_key in seen_keys or candidate_key in generation_keys:
                    continue
                generation_candidates.append(candidate)
                generation_keys.add(candidate_key)

            if not generation_candidates:
                state["status"] = "search_space_exhausted"
                break

            generation_rows = []
            generation_config_paths = []
            start_candidate_index = 1
            state["status"] = "running_generation"
            state["active_generation"] = generation_index
            state["active_generation_dir"] = str(generation_dir)
            state["active_candidates_completed"] = 0
            state["active_candidates_total"] = len(generation_candidates)
            state["active_population"] = copy.deepcopy(generation_candidates)
            state["active_generation_rows"] = []
            state["active_generation_config_paths"] = []

        _write_heartbeat(
            heartbeat_path,
            {
                **heartbeat,
                "status": "running_generation",
                "generation": generation_index,
                "active_generation": generation_index,
                "active_candidates_completed": int(state.get("active_candidates_completed", 0)),
                "active_candidates_total": len(generation_candidates),
                "candidates": len(generation_candidates),
                "generation_dir": str(generation_dir),
            },
        )
        dump_json(state_path, state)

        for candidate_index in range(start_candidate_index, len(generation_candidates) + 1):
            candidate = generation_candidates[candidate_index - 1]
            if resolved_deadline and _utc_now() >= resolved_deadline:
                state["status"] = "deadline_reached"
                generation_interrupted = True
                break
            if stop_path.exists():
                state["status"] = "stop_file_detected"
                generation_interrupted = True
                break

            if optimizer_kind == "config":
                payload = apply_config_overrides(base_payload, fixed_overrides)
                payload = apply_config_overrides(payload, candidate)
                base_name = str(payload.get("experiment_name", optimizer_name))
                experiment_name = f"{base_name}__g{generation_index:04d}__c{candidate_index:02d}"
                payload["experiment_name"] = experiment_name
                config_path = config_dir / f"{candidate_index:02d}_{slugify_name(payload['experiment_name'])}.json"
                dump_json(config_path, payload)
            else:
                experiment_name = f"{target_method}__g{generation_index:04d}__c{candidate_index:02d}"
                payload = _build_matrix_candidate_payload(
                    base_matrix_payload=base_payload,
                    target_method=target_method,
                    fixed_overrides=fixed_overrides,
                    candidate_params=candidate,
                )
                config_path = config_dir / f"{candidate_index:02d}_{slugify_name(experiment_name)}.json"
                dump_json(config_path, payload)
            row_fields: dict[str, object] = {
                "generation": generation_index,
                "candidate_index": candidate_index,
            }
            for key, value in candidate.items():
                row_fields[f"param_{slugify_name(key.replace('.', '_'))}"] = value
            generation_config_paths.append(config_path)

            if optimizer_kind == "config":
                row = _evaluate_study_item(
                    item={"config_path": config_path, "row_fields": row_fields},
                    index=candidate_index,
                    study_root=generation_dir,
                    mode=mode,
                    selection_metric=selection_metric,
                    metric_constraints=metric_constraints,
                    metric_report_names=metric_report_names,
                    seeds=spec.get("benchmark_seeds"),
                    walk_forward_folds=spec.get("walk_forward_folds"),
                    train_ratio_start=float(spec.get("train_ratio_start", 0.5)),
                )
            else:
                candidate_output_dir = generation_dir / f"{candidate_index:02d}_{slugify_name(experiment_name)}"
                row = _evaluate_matrix_candidate(
                    spec_path=config_path,
                    output_dir=candidate_output_dir,
                    experiment_name=experiment_name,
                    target_method=target_method,
                    selection_metric=selection_metric,
                    metric_constraints=metric_constraints,
                    metric_report_names=metric_report_names,
                    row_fields=row_fields,
                )
            generation_rows.append(row)

            candidate_result = {
                "generation": generation_index,
                "candidate_index": candidate_index,
                "candidate_key": _candidate_key(candidate),
                "params": copy.deepcopy(candidate),
                "selection_value": float(row["selection_value"]),
                "experiment": row["experiment"],
                "rank": 0,
                "summary_path": row["summary_path"],
                "constraint_feasible": bool(row.get("constraint_feasible", True)),
                "constraint_violation_score": float(row.get("constraint_violation_score", 0.0)),
                "constraint_violation_count": int(row.get("constraint_violation_count", 0)),
                "constraint_violations": list(row.get("constraint_violations", [])),
            }
            state["evaluated_candidates"].append(candidate_result)

            ranked_history = _rank_candidates(
                list(state["evaluated_candidates"]),
                selection_mode=selection_mode,
            )
            for rank, entry in enumerate(ranked_history, start=1):
                entry["rank"] = rank
            state["best_candidate"] = ranked_history[0] if ranked_history else None
            state["active_candidates_completed"] = candidate_index
            state["active_candidates_total"] = len(generation_candidates)
            state["active_generation_rows"] = copy.deepcopy(generation_rows)
            state["active_generation_config_paths"] = [str(path) for path in generation_config_paths]
            state["updated_at"] = _utc_now().isoformat()
            _write_progress_csv(progress_csv_path, list(state["evaluated_candidates"]))
            if optimizer_kind == "config":
                _write_best_snapshots(
                    best_config_path=best_config_path,
                    best_candidate_path=best_candidate_path,
                    base_payload=base_payload,
                    fixed_overrides=fixed_overrides,
                    best_candidate=state["best_candidate"],
                )
            else:
                _write_best_matrix_snapshots(
                    best_config_path=best_config_path,
                    best_candidate_path=best_candidate_path,
                    base_matrix_payload=base_payload,
                    target_method=target_method,
                    fixed_overrides=fixed_overrides,
                    best_candidate=state["best_candidate"],
                )
            dump_json(state_path, state)
            _write_heartbeat(
                heartbeat_path,
                {
                    "optimizer_name": optimizer_name,
                    "optimizer_kind": optimizer_kind,
                    "status": "running_generation",
                    "generation": generation_index,
                    "active_generation": generation_index,
                    "active_candidates_completed": candidate_index,
                    "active_candidates_total": len(generation_candidates),
                    "deadline": resolved_deadline.astimezone().isoformat() if resolved_deadline else None,
                    "updated_at": state["updated_at"],
                    "best_candidate": state["best_candidate"],
                    "stop_file": str(stop_path),
                    "generation_dir": str(generation_dir),
                },
            )

        if not generation_rows:
            break

        study_artifacts, study_summary = _finalize_study_outputs(
            study_root=generation_dir,
            rows=generation_rows,
            selection_metric=selection_metric,
            selection_mode=selection_mode,
            metric_constraints=metric_constraints,
            metric_report_names=metric_report_names,
            pareto_metrics=pareto_metrics,
            mode=mode,
            seeds=spec.get("benchmark_seeds"),
            walk_forward_folds=spec.get("walk_forward_folds"),
            train_ratio_start=float(spec.get("train_ratio_start", 0.5)),
            config_paths=generation_config_paths,
        )

        generation_record = {
            "generation": generation_index,
            "summary_path": str(study_artifacts.summary_path),
            "leaderboard_path": str(study_artifacts.leaderboard_csv_path),
            "top_experiment": study_summary["top_experiment"],
            "run_count": int(study_summary["run_count"]),
        }
        state["completed_generations"].append(generation_record)
        state["generation"] = generation_index
        rank_map = {
            (int(row["generation"]), int(row["candidate_index"])): int(row["rank"])
            for row in study_summary["leaderboard"]
        }
        for entry in state["evaluated_candidates"]:
            key = (int(entry["generation"]), int(entry["candidate_index"]))
            if key in rank_map:
                entry["rank"] = rank_map[key]
        if not generation_interrupted:
            state["status"] = "running"
        _clear_active_generation_state(state)
        dump_json(state_path, state)

    state["updated_at"] = _utc_now().isoformat()
    _write_progress_csv(progress_csv_path, list(state["evaluated_candidates"]))
    if optimizer_kind == "config":
        _write_best_snapshots(
            best_config_path=best_config_path,
            best_candidate_path=best_candidate_path,
            base_payload=base_payload,
            fixed_overrides=fixed_overrides,
            best_candidate=state["best_candidate"],
        )
    else:
        _write_best_matrix_snapshots(
            best_config_path=best_config_path,
            best_candidate_path=best_candidate_path,
            base_matrix_payload=base_payload,
            target_method=target_method,
            fixed_overrides=fixed_overrides,
            best_candidate=state["best_candidate"],
        )
    dump_json(state_path, state)
    _write_heartbeat(
        heartbeat_path,
        {
            "optimizer_name": optimizer_name,
            "optimizer_kind": optimizer_kind,
            "status": state["status"],
            "generation": int(state["generation"]),
            "active_generation": state.get("active_generation"),
            "active_candidates_completed": int(state.get("active_candidates_completed", 0)),
            "active_candidates_total": int(state.get("active_candidates_total", 0)),
            "deadline": resolved_deadline.astimezone().isoformat() if resolved_deadline else None,
            "updated_at": state["updated_at"],
            "best_candidate": state["best_candidate"],
            "stop_file": str(stop_path),
        },
    )
    return artifacts, state
