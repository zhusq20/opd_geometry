"""Exact-set allocation and response-clock state for the MOPD experiment."""

from __future__ import annotations

import copy
import hashlib
import itertools
import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize

TASKS = ("math", "code", "if", "science")
ALLOCATIONS = ("uniform", "gpas", "cost_gpas", "all")
ADAMW_STATES = ("conventional", "taskwise")
RUN_MODES = ("warm", "train", "bank")
FULL_PROMPTS = 16
RESPONSES_PER_PROMPT = 4
FULL_RESPONSES = FULL_PROMPTS * RESPONSES_PER_PROMPT
PROBE_PROMPTS = 2
PROBE_RESPONSES = PROBE_PROMPTS * RESPONSES_PER_PROMPT
MAIN_RESPONSE_BUDGET = 64_000
MAIN_CHECKPOINT_RESPONSES = (16_384, 32_768, 64_000)
MAIN_EVAL_RESPONSES = (2_048, 4_096, 8_192, 16_384, 32_768, 49_152, 64_000)
CONFIRMATION_RESPONSE_BUDGET = 16_384
CONFIRMATION_CHECKPOINT_RESPONSES = (16_384,)
CONFIRMATION_EVAL_RESPONSES = MAIN_EVAL_RESPONSES[:4]


def crossed_response_milestone(before: int, after: int, milestones: Sequence[int]) -> bool:
    """Return whether one configured response-clock milestone was crossed."""

    if int(after) < int(before):
        raise ValueError("the response clock cannot move backwards")
    return any(int(before) < int(value) <= int(after) for value in milestones)


def _finite(value: Any, name: str, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite and >= {minimum}, got {value!r}")
    return result


def task_sets(task_count: int, width: int) -> tuple[tuple[int, ...], ...]:
    if not 1 <= width <= task_count:
        raise ValueError(f"task width {width} is invalid for {task_count} tasks")
    return tuple(itertools.combinations(range(task_count), width))


def bounded_inclusion_probabilities(
    scores: Sequence[float], width: int, inclusion_floor: float
) -> list[float]:
    """Solve ``p_i=clip(score_i/z, floor, 1)`` with ``sum(p)=K``."""

    values = np.asarray([_finite(v, "score", 0.0) for v in scores], dtype=np.float64)
    count = int(values.size)
    floor = _finite(inclusion_floor, "inclusion_floor", 0.0)
    if not 1 <= int(width) <= count or count * floor > width + 1e-12:
        raise ValueError("infeasible exact-set inclusion bounds")
    if width == count:
        return [1.0] * count
    if not np.any(values > 0):
        return [float(width) / count] * count

    def total(z: float) -> float:
        return float(np.clip(values / z, floor, 1.0).sum())

    lo = 0.0
    hi = max(float(values.max()) / max(floor, 1e-12), 1.0)
    while total(hi) > width:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if total(mid) > width:
            lo = mid
        else:
            hi = mid
    result = np.clip(values / hi, floor, 1.0)
    free = np.flatnonzero((result > floor + 1e-10) & (result < 1.0 - 1e-10))
    residual = float(width - result.sum())
    if free.size:
        result[free] += residual / free.size
    else:
        result[int(np.argmax(1.0 - result))] += residual
    if not np.all((result >= floor - 1e-8) & (result <= 1.0 + 1e-8)):
        raise RuntimeError("bounded inclusion solver returned an infeasible result")
    result[-1] += width - float(result.sum())
    return result.tolist()


def inclusion_probabilities(
    sets: Sequence[Sequence[int]], distribution: Sequence[float], task_count: int
) -> list[float]:
    result = np.zeros(task_count, dtype=np.float64)
    for subset, probability in zip(sets, distribution, strict=True):
        result[list(subset)] += float(probability)
    return result.tolist()


def maximum_entropy_set_distribution(marginals: Sequence[float], width: int) -> dict[tuple[int, ...], float]:
    """Enumerate exact-K sets and find their maximum-entropy distribution."""

    p = np.asarray(marginals, dtype=np.float64)
    count = int(p.size)
    if not np.all(np.isfinite(p)) or np.any(p < -1e-10) or np.any(p > 1 + 1e-10):
        raise ValueError("invalid inclusion marginals")
    if not math.isclose(float(p.sum()), float(width), rel_tol=0, abs_tol=1e-8):
        raise ValueError(f"inclusion marginals sum to {p.sum()}, expected {width}")
    sets = task_sets(count, width)
    incidence = np.asarray([[int(i in subset) for i in range(count)] for subset in sets], dtype=np.float64)
    if len(sets) == 1:
        return {sets[0]: 1.0}

    # One natural parameter is redundant because every row sums to K. Fix the
    # final coordinate to zero and minimize the convex log-partition dual.
    def objective(theta_short: np.ndarray) -> tuple[float, np.ndarray]:
        theta = np.concatenate([theta_short, np.zeros(1)])
        logits = incidence @ theta
        shift = float(logits.max())
        weights = np.exp(logits - shift)
        q = weights / weights.sum()
        value = math.log(float(weights.sum())) + shift - float(p @ theta)
        gradient = incidence.T @ q - p
        return value, gradient[:-1]

    result = minimize(
        lambda x: objective(x)[0],
        np.zeros(count - 1, dtype=np.float64),
        jac=lambda x: objective(x)[1],
        method="BFGS",
        options={"gtol": 1e-12, "maxiter": 2000},
    )
    theta = np.concatenate([result.x, np.zeros(1)])
    logits = incidence @ theta
    logits -= logits.max()
    q = np.exp(logits)
    q /= q.sum()
    observed = incidence.T @ q
    if not np.allclose(observed, p, rtol=0, atol=2e-8):
        raise RuntimeError(f"maximum-entropy set solver missed marginals: {observed} != {p}")
    return {subset: float(probability) for subset, probability in zip(sets, q, strict=True)}


def execution_order(subset: Sequence[int], resident: int | None) -> tuple[int, ...]:
    selected = sorted(map(int, subset))
    if resident in selected:
        selected.remove(int(resident))
        selected.insert(0, int(resident))
    return tuple(selected)


def predicted_set_seconds(
    subset: Sequence[int],
    resident: int | None,
    task_seconds: Sequence[float],
    switch_seconds: Sequence[float],
) -> float:
    """Predict set critical-path time using task service and uncovered transfer tails."""

    order = execution_order(subset, resident)
    total = 0.0
    previous = resident
    for task in order:
        if previous != task:
            total += _finite(switch_seconds[task], "switch_seconds", 0.0)
        total += _finite(task_seconds[task], "task_seconds", 0.0)
        previous = task
    return max(total, 1e-12)


def cost_optimized_set_distribution(
    scores: Sequence[float],
    width: int,
    inclusion_floor: float,
    set_seconds: dict[tuple[int, ...], float],
) -> dict[tuple[int, ...], float]:
    """Directly minimize ``E[t(S)] * sum(a_i^2/p_i)`` over set probabilities."""

    a = np.asarray([_finite(v, "allocation score", 0.0) for v in scores], dtype=np.float64)
    sets = tuple(set_seconds)
    count = int(a.size)
    incidence = np.asarray([[int(i in subset) for i in range(count)] for subset in sets], dtype=np.float64)
    times = np.asarray([_finite(set_seconds[s], "set seconds", 1e-12) for s in sets], dtype=np.float64)
    floor = float(inclusion_floor)
    if width == count:
        return {sets[0]: 1.0}

    def objective(q: np.ndarray) -> float:
        p = incidence.T @ q
        return float((times @ q) * np.sum(a * a / np.maximum(p, 1e-15)))

    constraints: list[dict[str, Any]] = [{"type": "eq", "fun": lambda q: float(q.sum() - 1.0)}]
    for task in range(count):
        constraints.append({"type": "ineq", "fun": lambda q, i=task: float(incidence[:, i] @ q - floor)})
        constraints.append({"type": "ineq", "fun": lambda q, i=task: float(1.0 - incidence[:, i] @ q)})
    result = minimize(
        objective,
        np.full(len(sets), 1.0 / len(sets), dtype=np.float64),
        method="SLSQP",
        bounds=[(1e-12, 1.0)] * len(sets),
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    q = np.maximum(result.x, 0.0)
    q /= q.sum()
    p = incidence.T @ q
    if not result.success or np.any(p < floor - 2e-7) or np.any(p > 1 + 2e-7):
        raise RuntimeError(f"set-aware Cost-GPAS solver failed: {result.message}; p={p}")
    return {subset: float(probability) for subset, probability in zip(sets, q, strict=True)}


@dataclass(frozen=True)
class ControllerConfig:
    task_names: tuple[str, ...]
    target_weights: tuple[float, ...]
    allocation: str
    task_width: int
    adamw_state: str
    run_mode: str
    seed: int
    rollout_offset: int
    ema_decay: float
    inclusion_floor: float
    score_max_age: int
    response_budget: int
    checkpoint_responses: tuple[int, ...]
    bank_units_per_task: int


class MOPDController:
    """Plan exact task sets, counted probes, and their serializable clocks."""

    SCHEMA_VERSION = 3

    def __init__(
        self,
        task_names: Sequence[str] = TASKS,
        *,
        target_weights: Sequence[float] | None = None,
        allocation: str = "uniform",
        task_width: int = 1,
        adamw_state: str = "taskwise",
        run_mode: str = "train",
        seed: int = 42,
        rollout_offset: int = 0,
        ema_decay: float = 0.95,
        inclusion_floor: float = 0.05,
        score_max_age: int = 50,
        response_budget: int = MAIN_RESPONSE_BUDGET,
        checkpoint_responses: Sequence[int] = MAIN_CHECKPOINT_RESPONSES,
        bank_units_per_task: int = 8,
    ) -> None:
        names = tuple(map(str, task_names))
        if names != TASKS:
            raise ValueError(f"task order must be {TASKS}, got {names}")
        weights = tuple(float(v) for v in (target_weights or [0.25] * len(names)))
        if len(weights) != len(names) or not math.isclose(sum(weights), 1.0, abs_tol=1e-12):
            raise ValueError("target weights must contain four values summing to one")
        if allocation not in ALLOCATIONS or adamw_state not in ADAMW_STATES or run_mode not in RUN_MODES:
            raise ValueError("unknown MOPD controller configuration")
        if int(task_width) not in {1, 2, 4}:
            raise ValueError("the frozen MOPD protocol supports task widths 1, 2, and 4")
        if allocation == "all" and task_width != len(names):
            raise ValueError("allocation=all requires K=4")
        if task_width == len(names) and allocation != "all" and run_mode == "train":
            raise ValueError("K=4 training uses allocation=all")
        decay = _finite(ema_decay, "ema_decay", 0.0)
        if decay >= 1:
            raise ValueError("ema_decay must be below one")
        bounded_inclusion_probabilities([1.0] * len(names), int(task_width), inclusion_floor)
        if int(response_budget) <= 0 or int(response_budget) % PROBE_RESPONSES:
            raise ValueError("the response budget must be a positive multiple of eight")
        milestones = tuple(sorted(set(map(int, checkpoint_responses))))
        if any(v <= 0 or v > int(response_budget) or v % PROBE_RESPONSES for v in milestones):
            raise ValueError("checkpoint response clocks must be positive multiples of eight within the budget")
        self.config = ControllerConfig(
            task_names=names,
            target_weights=weights,
            allocation=allocation,
            task_width=int(task_width),
            adamw_state=adamw_state,
            run_mode=run_mode,
            seed=int(seed),
            rollout_offset=int(rollout_offset),
            ema_decay=decay,
            inclusion_floor=float(inclusion_floor),
            score_max_age=int(score_max_age),
            response_budget=int(response_budget),
            checkpoint_responses=milestones,
            bank_units_per_task=int(bank_units_per_task),
        )
        self.rng = random.Random(self.config.seed)
        n = len(names)
        self.raw_gradient_sq_ema = [0.0] * n
        self.adam_gradient_sq_ema = [0.0] * n
        self.task_seconds_ema = [0.0] * n
        self.switch_seconds_ema = [0.0] * n
        self.observation_counts = [0] * n
        self.switch_observation_counts = [0] * n
        self.score_ages = [0] * n
        self.processed_task_units = 0
        self.attempted_responses = 0
        self.completed_operations = 0
        self.optimizer_updates = 0
        self.probe_count = 0
        self.resident_teacher: int | None = 0
        self.pending: dict[str, Any] | None = None
        self._budget_fill_cursor = 0

    @property
    def task_names(self) -> tuple[str, ...]:
        return self.config.task_names

    @property
    def budget_complete(self) -> bool:
        if self.config.run_mode == "bank":
            return self.completed_operations >= self.config.bank_units_per_task * len(self.task_names)
        if self.config.run_mode == "warm":
            return self.completed_operations >= 2 * len(self.task_names)
        return self.attempted_responses >= self.config.response_budget

    def _bias_corrected(
        self, values: Sequence[float], counts: Sequence[int] | None = None
    ) -> list[float | None]:
        """Return task-local EMA estimates on a common, unbiased scale."""

        decay = self.config.ema_decay
        counts = self.observation_counts if counts is None else counts
        return [
            float(value) / (1.0 - decay**count) if count else None
            for value, count in zip(values, counts, strict=True)
        ]

    def _effective(
        self, values: Sequence[float], default: float, counts: Sequence[int] | None = None
    ) -> list[float]:
        corrected = self._bias_corrected(values, counts)
        observed = [value for value in corrected if value is not None]
        fallback = float(sum(observed) / len(observed)) if observed else default
        return [fallback if value is None else float(value) for value in corrected]

    def _score_rms(self, values: Sequence[float]) -> np.ndarray:
        corrected = [0.0 if value is None else value for value in self._bias_corrected(values)]
        return np.sqrt(np.maximum(corrected, 0.0))

    def _proposal(self) -> tuple[list[float], dict[tuple[int, ...], float], dict[tuple[int, ...], float]]:
        n, width = len(self.task_names), self.config.task_width
        sets = task_sets(n, width)
        adam = self._score_rms(self.adam_gradient_sq_ema)
        target = np.asarray(self.config.target_weights)
        task_seconds = self._effective(self.task_seconds_ema, 1.0)
        switch_seconds = self._effective(
            self.switch_seconds_ema, 0.0, self.switch_observation_counts
        )
        costs = {
            subset: predicted_set_seconds(subset, self.resident_teacher, task_seconds, switch_seconds)
            for subset in sets
        }
        if self.config.allocation == "all":
            marginals = [1.0] * n
            distribution = {sets[0]: 1.0}
        elif self.config.allocation == "uniform":
            marginals = [width / n] * n
            distribution = {subset: 1.0 / len(sets) for subset in sets}
        elif self.config.allocation == "gpas":
            marginals = bounded_inclusion_probabilities(target * adam, width, self.config.inclusion_floor)
            distribution = maximum_entropy_set_distribution(marginals, width)
        else:
            distribution = cost_optimized_set_distribution(
                target * adam, width, self.config.inclusion_floor, costs
            )
            marginals = inclusion_probabilities(tuple(distribution), tuple(distribution.values()), n)
        return list(map(float, marginals)), distribution, costs

    def _draw(self, distribution: dict[tuple[int, ...], float]) -> tuple[int, ...]:
        threshold = self.rng.random()
        cumulative = 0.0
        for subset, probability in distribution.items():
            cumulative += probability
            if threshold < cumulative:
                return subset
        return next(reversed(distribution))

    def _next_milestone(self) -> int:
        for value in self.config.checkpoint_responses:
            if value > self.attempted_responses:
                return value
        return self.config.response_budget

    def _probe_task(self) -> tuple[int, str]:
        stale = [
            i
            for i, age in enumerate(self.score_ages)
            if age + self.config.task_width > self.config.score_max_age
        ]
        if stale:
            return max(stale, key=lambda i: (self.score_ages[i], -i)), "stale_score"
        task = self._budget_fill_cursor % len(self.task_names)
        self._budget_fill_cursor += 1
        return task, "response_clock_alignment"

    def plan(self, rollout_id: int) -> dict[str, Any]:
        if self.pending is not None:
            if int(self.pending["rollout_id"]) == int(rollout_id):
                return copy.deepcopy(self.pending)
            raise RuntimeError("the previous MOPD operation still awaits feedback")
        expected = self.config.rollout_offset + self.completed_operations
        if int(rollout_id) != expected:
            raise ValueError(f"expected rollout id {expected}, got {rollout_id}")
        if self.budget_complete:
            raise StopIteration("MOPD attempted-response budget is complete")

        if self.config.run_mode == "bank":
            task = self.completed_operations % len(self.task_names)
            operation, reason = "bank", "frozen_bank"
            selected, prompt_count, step_size = (task,), FULL_PROMPTS, FULL_RESPONSES
            marginals = [0.0] * len(self.task_names)
            marginals[task] = 1.0
            distribution = {(task,): 1.0}
            costs = {(task,): self._effective(self.task_seconds_ema, 1.0)[task]}
        elif self.config.run_mode == "warm":
            task = self.completed_operations % len(self.task_names)
            operation, reason = "train", "round_robin_warm_start"
            selected, prompt_count, step_size = (task,), FULL_PROMPTS, FULL_RESPONSES
            marginals = [0.25] * len(self.task_names)
            distribution = {(i,): 0.25 for i in range(len(self.task_names))}
            costs = {(i,): self._effective(self.task_seconds_ema, 1.0)[i] for i in range(len(self.task_names))}
        else:
            train_responses = FULL_RESPONSES * self.config.task_width
            remaining_to_boundary = self._next_milestone() - self.attempted_responses
            stale = self.config.allocation in {"gpas", "cost_gpas"} and any(
                age + self.config.task_width > self.config.score_max_age
                for age in self.score_ages
            )
            if stale or remaining_to_boundary < train_responses:
                task, reason = self._probe_task()
                operation, selected = "probe", (task,)
                prompt_count, step_size = PROBE_PROMPTS, RESPONSES_PER_PROMPT
                marginals = [0.0] * len(self.task_names)
                marginals[task] = 1.0
                distribution = {(task,): 1.0}
                costs = {(task,): self._effective(self.task_seconds_ema, 1.0)[task]}
            else:
                if any(c < 2 for c in self.observation_counts):
                    raise RuntimeError("training started without two warm-start observations per task")
                operation, reason = "train", "allocated_exact_set"
                prompt_count, step_size = FULL_PROMPTS * self.config.task_width, FULL_RESPONSES
                marginals, distribution, costs = self._proposal()
                selected = self._draw(distribution)

        order = execution_order(selected, self.resident_teacher)
        per_task = []
        for task in order:
            probability = float(marginals[task])
            target = float(self.config.target_weights[task])
            per_task.append(
                {
                    "task_index": task,
                    "task": self.task_names[task],
                    "inclusion_probability": probability,
                    "target_weight": target,
                    "importance_correction": target / probability,
                }
            )
        attempted = PROBE_RESPONSES if operation == "probe" else FULL_RESPONSES * len(selected)
        self.pending = {
            "schema_version": self.SCHEMA_VERSION,
            "rollout_id": int(rollout_id),
            "operation_index": self.completed_operations,
            "operation": operation,
            "reason": reason,
            "run_mode": self.config.run_mode,
            "allocation": self.config.allocation,
            "adamw_state": self.config.adamw_state,
            "task_width": len(selected),
            "selected_set": [self.task_names[i] for i in selected],
            "selected_indices": list(selected),
            "execution_order": [self.task_names[i] for i in order],
            "execution_indices": list(order),
            "resident_teacher_before": None if self.resident_teacher is None else self.task_names[self.resident_teacher],
            "inclusion_probabilities": dict(zip(self.task_names, marginals, strict=True)),
            "set_distribution": {"+".join(self.task_names[i] for i in subset): q for subset, q in distribution.items()},
            "predicted_set_seconds": {"+".join(self.task_names[i] for i in subset): costs[subset] for subset in costs},
            "task_units": per_task,
            "prompt_count": prompt_count,
            "responses_per_prompt": RESPONSES_PER_PROMPT,
            "attempted_responses": attempted,
            "attempted_responses_before": self.attempted_responses,
            "processed_task_units_before": self.processed_task_units,
            "optimizer_updates_before": self.optimizer_updates,
            "score_ages_before": dict(zip(self.task_names, self.score_ages, strict=True)),
            "raw_gradient_rms_before": dict(zip(self.task_names, self._score_rms(self.raw_gradient_sq_ema), strict=True)),
            "adam_gradient_rms_before": dict(zip(self.task_names, self._score_rms(self.adam_gradient_sq_ema), strict=True)),
            "task_seconds_before": dict(zip(self.task_names, self._effective(self.task_seconds_ema, 1.0), strict=True)),
            "switch_seconds_before": dict(
                zip(
                    self.task_names,
                    self._effective(
                        self.switch_seconds_ema, 0.0, self.switch_observation_counts
                    ),
                    strict=True,
                )
            ),
            "step_global_batch_size": step_size,
            "rng_state_sha256_before": hashlib.sha256(repr(self.rng.getstate()).encode()).hexdigest(),
            "issued": 0,
        }
        return copy.deepcopy(self.pending)

    def complete(self, rollout_id: int, feedback: dict[str, Any]) -> dict[str, Any]:
        if self.pending is None or int(self.pending["rollout_id"]) != int(rollout_id):
            raise RuntimeError(f"no pending operation for rollout {rollout_id}")
        pending = self.pending
        if feedback.get("operation") != pending["operation"]:
            raise ValueError("trainer feedback operation differs from the allocation plan")
        units = list(feedback.get("task_units") or [])
        if [unit["task"] for unit in units] != pending["execution_order"]:
            raise ValueError("trainer feedback task order differs from the streamed teacher order")
        if int(feedback["attempted_responses"]) != int(pending["attempted_responses"]):
            raise ValueError("attempted-response accounting differs from the allocation plan")

        decay = self.config.ema_decay
        for unit in units:
            index = self.task_names.index(str(unit["task"]))
            for vector, key in (
                (self.raw_gradient_sq_ema, "raw_score"),
                (self.adam_gradient_sq_ema, "adam_score"),
            ):
                score = _finite(unit[key], key, 0.0)
                vector[index] = decay * vector[index] + (1 - decay) * score * score
            task_seconds = _finite(unit["predicted_full_task_seconds"], "predicted_full_task_seconds", 0.0)
            # Weight loading overlaps the corresponding student rollout. The
            # proposal uses only the uncovered critical-path tail; raw switch
            # duration remains available in the system trace.
            self.task_seconds_ema[index] = decay * self.task_seconds_ema[index] + (1 - decay) * task_seconds
            if bool(unit["switched"]):
                switch_seconds = _finite(
                    unit["teacher_transfer_tail_seconds"], "teacher_transfer_tail_seconds", 0.0
                )
                self.switch_seconds_ema[index] = (
                    decay * self.switch_seconds_ema[index] + (1 - decay) * switch_seconds
                )
                self.switch_observation_counts[index] += 1
            self.observation_counts[index] += 1
            self.score_ages[index] = 0

        if pending["operation"] == "train":
            processed = len(pending["selected_indices"])
            self.score_ages = [age + processed for age in self.score_ages]
            for index in pending["selected_indices"]:
                self.score_ages[int(index)] = 0
            self.processed_task_units += processed
            self.optimizer_updates += 1
        elif pending["operation"] == "probe":
            self.probe_count += 1
        self.attempted_responses += int(pending["attempted_responses"])
        self.completed_operations += 1
        resident = feedback.get("resident_teacher_after")
        self.resident_teacher = None if resident is None else self.task_names.index(str(resident))

        record = copy.deepcopy(pending)
        record.pop("issued", None)
        record.update(
            {
                "feedback": copy.deepcopy(feedback),
                "attempted_responses_after": self.attempted_responses,
                "processed_task_units_after": self.processed_task_units,
                "optimizer_updates_after": self.optimizer_updates,
                "probe_count_after": self.probe_count,
                "resident_teacher_after": resident,
                "score_ages_after": dict(zip(self.task_names, self.score_ages, strict=True)),
                "raw_gradient_rms_after": dict(zip(self.task_names, self._score_rms(self.raw_gradient_sq_ema), strict=True)),
                "adam_gradient_rms_after": dict(zip(self.task_names, self._score_rms(self.adam_gradient_sq_ema), strict=True)),
                "task_seconds_after": dict(zip(self.task_names, self._effective(self.task_seconds_ema, 1.0), strict=True)),
                "switch_seconds_after": dict(
                    zip(
                        self.task_names,
                        self._effective(
                            self.switch_seconds_ema, 0.0, self.switch_observation_counts
                        ),
                        strict=True,
                    )
                ),
                "checkpoint_due": self.attempted_responses in self.config.checkpoint_responses,
                "budget_complete": self.budget_complete,
                "rng_state_sha256_after": hashlib.sha256(repr(self.rng.getstate()).encode()).hexdigest(),
            }
        )
        self.pending = None
        return record

    def bootstrap(self, state: dict[str, Any]) -> None:
        """Copy warm/checkpoint statistics into a new experiment branch."""

        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError("bootstrap controller state has the wrong schema")
        if tuple(state["config"]["task_names"]) != self.task_names:
            raise ValueError("bootstrap controller uses a different task order")
        for key in (
            "raw_gradient_sq_ema", "adam_gradient_sq_ema", "task_seconds_ema",
            "switch_seconds_ema", "observation_counts", "switch_observation_counts", "score_ages",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
        self.processed_task_units = int(state["processed_task_units"])
        self.attempted_responses = int(state["attempted_responses"])
        self.optimizer_updates = int(state["optimizer_updates"])
        self.resident_teacher = state.get("resident_teacher")
        self.completed_operations = 0
        self.probe_count = 0
        self.pending = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "config": asdict(self.config),
            "rng_state": self.rng.getstate(),
            "raw_gradient_sq_ema": self.raw_gradient_sq_ema,
            "adam_gradient_sq_ema": self.adam_gradient_sq_ema,
            "task_seconds_ema": self.task_seconds_ema,
            "switch_seconds_ema": self.switch_seconds_ema,
            "observation_counts": self.observation_counts,
            "switch_observation_counts": self.switch_observation_counts,
            "score_ages": self.score_ages,
            "processed_task_units": self.processed_task_units,
            "attempted_responses": self.attempted_responses,
            "completed_operations": self.completed_operations,
            "optimizer_updates": self.optimizer_updates,
            "probe_count": self.probe_count,
            "resident_teacher": self.resident_teacher,
            "budget_fill_cursor": self._budget_fill_cursor,
            "pending": self.pending,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if int(state.get("schema_version", -1)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported MOPD controller checkpoint schema")
        config = dict(state["config"])
        for key in ("task_names", "target_weights", "checkpoint_responses"):
            config[key] = tuple(config[key])
        if config != asdict(self.config):
            raise ValueError("saved controller configuration differs from this run")
        self.rng.setstate(state["rng_state"])
        for key in (
            "raw_gradient_sq_ema", "adam_gradient_sq_ema", "task_seconds_ema",
            "switch_seconds_ema", "observation_counts", "switch_observation_counts", "score_ages",
        ):
            setattr(self, key, copy.deepcopy(state[key]))
        for key in (
            "processed_task_units", "attempted_responses", "completed_operations",
            "optimizer_updates", "probe_count",
        ):
            setattr(self, key, int(state[key]))
        self.resident_teacher = state.get("resident_teacher")
        self._budget_fill_cursor = int(state.get("budget_fill_cursor", 0))
        self.pending = copy.deepcopy(state.get("pending"))


__all__ = [
    "ADAMW_STATES", "ALLOCATIONS", "CONFIRMATION_CHECKPOINT_RESPONSES",
    "CONFIRMATION_EVAL_RESPONSES", "CONFIRMATION_RESPONSE_BUDGET", "FULL_PROMPTS",
    "FULL_RESPONSES", "MAIN_CHECKPOINT_RESPONSES", "MAIN_EVAL_RESPONSES",
    "MAIN_RESPONSE_BUDGET", "MOPDController", "PROBE_PROMPTS", "PROBE_RESPONSES",
    "RESPONSES_PER_PROMPT", "RUN_MODES", "TASKS", "bounded_inclusion_probabilities",
    "cost_optimized_set_distribution", "execution_order", "inclusion_probabilities",
    "crossed_response_milestone", "maximum_entropy_set_distribution", "predicted_set_seconds",
    "task_sets",
]
