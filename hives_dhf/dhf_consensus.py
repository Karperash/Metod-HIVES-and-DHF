from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Tuple

import numpy as np


Method = Literal["GA", "HHO"]


@dataclass(frozen=True)
class DHFOptimizationResult:
    dm_ids: List[str]
    best_weights: np.ndarray  # shape (num_dms,), sum=1
    best_consensus: float
    compatibility: List[float]
    method: Method


def _normalize_weights(w: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    w[w < 0] = 0.0
    s = float(w.sum())
    if s <= eps:
        w[:] = 1.0 / len(w)
        return w
    return w / s


def _clamp_and_normalize(
    w: np.ndarray,
    min_w: float = 0.01,
    max_w: float = 0.99,
    eps: float = 1e-12,
) -> np.ndarray:
    w = np.asarray(w, dtype=float).copy()
    w = np.clip(w, min_w, max_w)
    s = float(w.sum())
    if s <= eps:
        w[:] = 1.0 / len(w)
        return w
    return w / s


def load_dhf_payload(data: Dict[str, Any]) -> Tuple[List[str], List[str], Dict[str, Any]]:
    """
    Validates and extracts DHF payload.

    Expected structure (as in программа/input_data.json / main4.py):
      - criteria: List[str]
      - dms: List[{id, pairwise_comparisons: {...}}]
      - parameters: optional
    """
    criteria = list(data["criteria"])
    dms = list(data["dms"])
    dm_ids = [str(dm["id"]) for dm in dms]
    return criteria, dm_ids, data


def dhf_json_to_matrices(
    data: Dict[str, Any],
    max_dhf_elements: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Converts DHF JSON to 4D matrices compatible with compat3():
      Judgments_membership[i, k, j, dm]
      Judgments_nonmembership[i, k, j, dm]
    """
    criteria, dm_ids, payload = load_dhf_payload(data)
    dms_data = payload["dms"]

    num_criteria = len(criteria)
    num_dms = len(dms_data)

    jm = np.zeros((num_criteria, max_dhf_elements, num_criteria, num_dms), dtype=float)
    jn = np.zeros((num_criteria, max_dhf_elements, num_criteria, num_dms), dtype=float)

    for dm_idx, dm in enumerate(dms_data):
        comparisons = dm["pairwise_comparisons"]
        for i, crit1 in enumerate(criteria):
            for j, crit2 in enumerate(criteria):
                comp_data = comparisons[crit1][crit2]

                membership_vals = list(comp_data["membership"])
                for k in range(max_dhf_elements):
                    jm[i, k, j, dm_idx] = membership_vals[k] if k < len(membership_vals) else 0.0

                nonmembership_vals = list(comp_data["non_membership"])
                for k in range(max_dhf_elements):
                    jn[i, k, j, dm_idx] = nonmembership_vals[k] if k < len(nonmembership_vals) else 0.0

    return jm, jn, criteria, dm_ids


def compat3(dm_weights: np.ndarray, judgments_m: np.ndarray, judgments_n: np.ndarray) -> List[float]:
    """
    Computes compatibility scores for each DM given DM weights.
    This is a direct port of the logic from main4.py, kept intentionally similar.
    """
    W = np.asarray(dm_weights, dtype=float)
    num_criteria = judgments_m.shape[0]
    max_dhf = judgments_m.shape[1]
    num_dms = judgments_m.shape[3]

    if len(W) != num_dms:
        raise ValueError(f"DM weights length ({len(W)}) must match num_dms ({num_dms})")

    # Precompute combinations of DHF indices per DM
    import itertools

    all_combinations = list(itertools.product(*[range(max_dhf) for _ in range(num_dms)]))

    # MEMBERSHIP aggregation
    DHFWHM = np.zeros((num_criteria, num_criteria))
    sum_count = np.zeros((num_criteria, num_criteria))

    for i in range(num_criteria):
        for j in range(num_criteria):
            valid = 0
            for comb in all_combinations:
                # require all non-zero
                if any(judgments_m[i, comb[dm], j, dm] == 0 for dm in range(num_dms)):
                    continue

                term1 = 1.0
                term2 = 1.0
                for dm in range(num_dms):
                    val = judgments_m[i, comb[dm], j, dm]
                    term1 *= val ** W[dm]
                    term2 *= (1 - val) ** W[dm]

                denom = term1 + term2
                if denom != 0:
                    DHFWHM[i, j] += term1 / denom
                    valid += 1
            sum_count[i, j] = valid

    membership = np.zeros((num_criteria, num_criteria))
    for i in range(num_criteria):
        for j in range(num_criteria):
            membership[i, j] = DHFWHM[i, j] / sum_count[i, j] if sum_count[i, j] > 0 else 0.5

    # NON-MEMBERSHIP aggregation
    DHFWHM2 = np.zeros((num_criteria, num_criteria))
    sum_count2 = np.zeros((num_criteria, num_criteria))

    for i in range(num_criteria):
        for j in range(num_criteria):
            valid = 0
            for comb in all_combinations:
                if any(judgments_n[i, comb[dm], j, dm] == 0 for dm in range(num_dms)):
                    continue

                term1 = 1.0
                term2 = 1.0
                for dm in range(num_dms):
                    val = judgments_n[i, comb[dm], j, dm]
                    term1 *= val ** W[dm]
                    term2 *= (1 - val) ** W[dm]
                denom = term1 + term2
                if denom != 0:
                    DHFWHM2[i, j] += term1 / denom
                    valid += 1
                sum_count2[i, j] = valid

    nonmembership = np.zeros((num_criteria, num_criteria))
    for i in range(num_criteria):
        for j in range(num_criteria):
            nonmembership[i, j] = DHFWHM2[i, j] / sum_count2[i, j] if sum_count2[i, j] > 0 else 0.5

    # COMPATIBILITY per DM
    compatibility: List[float] = [0.0] * num_dms
    for dm_idx in range(num_dms):
        calculus2 = 0.0

        # Membership part
        value = 0.0
        x = 0
        for k in range(num_criteria):
            for l in range(num_criteria):
                for m in range(max_dhf):
                    v = judgments_m[k, m, l, dm_idx]
                    if v != 0:
                        value += v * membership[k, l]
                        x += 1
                if x > 0:
                    calculus2 += value / x
                    value = 0.0
                    x = 0

        # Non-membership part
        value = 0.0
        x = 0
        for k in range(num_criteria):
            for l in range(num_criteria):
                for m in range(max_dhf):
                    v = judgments_n[k, m, l, dm_idx]
                    if v != 0:
                        value += v * nonmembership[k, l]
                        x += 1
                if x > 0:
                    calculus2 += value / x
                    value = 0.0
                    x = 0

        # numerator extras
        t = 0
        r = 0
        rest1 = 0.0
        rest11 = 0.0
        rest2 = 0.0
        rest22 = 0.0

        for k in range(num_criteria):
            for l in range(num_criteria):
                for m in range(max_dhf):
                    if judgments_m[k, m, l, dm_idx] != 0:
                        t += 1
                        rest1 += judgments_m[k, m, l, dm_idx]
                    if judgments_n[k, m, l, dm_idx] != 0:
                        r += 1
                        rest11 += judgments_n[k, m, l, dm_idx]

                rest2 += membership[k, l]
                rest22 += nonmembership[k, l]

        restf1 = 0.0
        if t > 0 and r > 0:
            restf1 = 1 - (rest1 / t) - (rest11 / r)
        restf2 = 1 - rest2 - rest22

        upside = (restf1 * restf2) + calculus2

        # denominator
        pertalone1 = 0.0
        npertalone1 = 0.0
        pertalone2 = 0.0
        npertalone2 = 0.0
        t = 0
        r = 0

        for k in range(num_criteria):
            for l in range(num_criteria):
                for m in range(max_dhf):
                    if judgments_m[k, m, l, dm_idx] != 0:
                        t += 1
                        pertalone1 += judgments_m[k, m, l, dm_idx] ** 2
                    if judgments_n[k, m, l, dm_idx] != 0:
                        r += 1
                        npertalone1 += judgments_n[k, m, l, dm_idx] ** 2

                pertalone2 += membership[k, l] ** 2
                npertalone2 += nonmembership[k, l] ** 2

        pertalone11 = (pertalone1 / t) if t > 0 else 0.0
        npertalone11 = (npertalone1 / r) if r > 0 else 0.0

        restff1 = restf1 ** 2
        restff2 = restf2 ** 2

        denom = (np.sqrt(pertalone11 + npertalone11 + restff1) * np.sqrt(pertalone2 + npertalone2 + restff2))
        compatibility[dm_idx] = float(upside / denom) if denom != 0 else 0.0

    return compatibility


def genetic_algorithm_optimize(
    judgments_m: np.ndarray,
    judgments_n: np.ndarray,
    desired_consensus: float = 0.907,
    population_size: int = 20,
    max_iterations: int = 500,
    min_w: float = 0.01,
    max_w: float = 0.99,
) -> Tuple[np.ndarray, float, List[float]]:
    num_dms = judgments_m.shape[3]

    best_weights = np.ones(num_dms) / num_dms
    best_fitness = -1.0
    best_compat: List[float] = []

    population: List[np.ndarray] = []
    for _ in range(population_size):
        w = np.random.random(num_dms)
        population.append(_clamp_and_normalize(w, min_w=min_w, max_w=max_w))

    for _it in range(max_iterations):
        fitness_scores: List[float] = []

        for w in population:
            comp = compat3(w, judgments_m, judgments_n)
            fitness = float(min(comp))
            fitness_scores.append(fitness)
            if fitness > best_fitness:
                best_fitness = fitness
                best_weights = w.copy()
                best_compat = list(comp)

        if best_fitness >= desired_consensus:
            break

        # elite selection
        elite_size = max(1, population_size // 5)
        elite_indices = np.argsort(fitness_scores)[-elite_size:]
        new_population = [population[idx] for idx in elite_indices]

        # crossover + mutation
        while len(new_population) < population_size:
            parent1 = random.choice(population)
            parent2 = random.choice(population)
            if num_dms == 1:
                child = parent1.copy()
            else:
                cp = random.randint(1, num_dms - 1)
                child = np.concatenate([parent1[:cp], parent2[cp:]])
            # mutation
            if random.random() < 0.3 and num_dms >= 1:
                pos = random.randint(0, num_dms - 1)
                child[pos] = random.random()
            child = _clamp_and_normalize(child, min_w=min_w, max_w=max_w)
            new_population.append(child)

        population = new_population

    return best_weights, best_fitness, best_compat


def hho_optimize(
    judgments_m: np.ndarray,
    judgments_n: np.ndarray,
    desired_consensus: float = 0.907,
    search_agents_no: int = 10,
    max_iter: int = 100,
    min_w: float = 0.01,
    max_w: float = 0.99,
) -> Tuple[np.ndarray, float, List[float]]:
    num_dms = judgments_m.shape[3]
    dim = num_dms
    lb = np.full(dim, min_w, dtype=float)
    ub = np.full(dim, max_w, dtype=float)

    best_weights = np.ones(num_dms) / num_dms
    best_fitness = -1.0
    best_compat: List[float] = []

    X = np.zeros((search_agents_no, dim), dtype=float)
    for i in range(search_agents_no):
        X[i] = _clamp_and_normalize(np.random.random(dim), min_w=min_w, max_w=max_w)

    rabbit_location = best_weights.copy()
    rabbit_energy = float("inf")

    fitness_values = np.full(search_agents_no, float("inf"), dtype=float)

    for t in range(max_iter):
        for i in range(search_agents_no):
            w = _clamp_and_normalize(X[i], min_w=min_w, max_w=max_w)
            comp = compat3(w, judgments_m, judgments_n)
            current_cons = float(min(comp))
            loss = 1.0 - current_cons
            fitness_values[i] = loss
            if current_cons >= best_fitness:
                best_fitness = current_cons
                best_weights = w.copy()
                best_compat = list(comp)

        min_idx = int(np.argmin(fitness_values))
        if fitness_values[min_idx] < rabbit_energy:
            rabbit_energy = float(fitness_values[min_idx])
            rabbit_location = X[min_idx].copy()

        if best_fitness >= desired_consensus:
            break

        E1 = 2 * (1 - t / max_iter)
        for i in range(search_agents_no):
            E0 = 2 * np.random.random() - 1
            escaping_energy = E1 * E0

            if abs(escaping_energy) >= 1:
                q = np.random.random()
                if q < 0.5:
                    rand_idx = np.random.randint(0, search_agents_no)
                    X_rand = X[rand_idx]
                    X[i] = X_rand - np.random.random() * np.abs(X_rand - 2 * np.random.random() * X[i])
                else:
                    X[i] = (rabbit_location - X.mean(axis=0)) - np.random.random() * (ub - lb) * np.random.random() + lb
            else:
                r = np.random.random()
                if r >= 0.5 and abs(escaping_energy) < 0.5:
                    X[i] = rabbit_location - escaping_energy * np.abs(rabbit_location - X[i])
                elif r >= 0.5:
                    jump = 2 * (1 - np.random.random())
                    X[i] = rabbit_location + jump * (X[i] - rabbit_location)
                else:
                    jump = 2 * np.random.random()
                    X1 = rabbit_location - escaping_energy * np.abs(jump * rabbit_location - X[i])
                    w1 = _clamp_and_normalize(X1, min_w=min_w, max_w=max_w)
                    try:
                        comp1 = compat3(w1, judgments_m, judgments_n)
                        fit1 = 1.0 - float(min(comp1))
                    except Exception:
                        fit1 = float("inf")
                    if fit1 < fitness_values[i]:
                        X[i] = X1
                    else:
                        X2 = X1 + np.random.randn(dim) * 0.1
                        w2 = _clamp_and_normalize(X2, min_w=min_w, max_w=max_w)
                        try:
                            comp2 = compat3(w2, judgments_m, judgments_n)
                            fit2 = 1.0 - float(min(comp2))
                        except Exception:
                            fit2 = float("inf")
                        if fit2 < fitness_values[i]:
                            X[i] = X2

            X[i] = np.clip(X[i], lb, ub)
            X[i] = _clamp_and_normalize(X[i], min_w=min_w, max_w=max_w)

    return best_weights, best_fitness, best_compat


def optimize_expert_weights(
    dhf_data: Dict[str, Any],
    method: Method = "HHO",
    desired_consensus: float | None = None,
    population_size: int | None = None,
    max_iterations: int | None = None,
    search_agents_no: int | None = None,
    max_iter: int | None = None,
    min_w: float = 0.01,
    max_w: float = 0.99,
) -> DHFOptimizationResult:
    """
    High-level API: given DHF JSON, optimize DM weights using GA or HHO.
    """
    jm, jn, _criteria, dm_ids = dhf_json_to_matrices(dhf_data)

    params = dict(dhf_data.get("parameters") or {})
    desired = float(desired_consensus if desired_consensus is not None else params.get("desired_consensus", 0.907))

    if method == "GA":
        pop = int(population_size if population_size is not None else params.get("population_size", 20))
        iters = int(max_iterations if max_iterations is not None else params.get("max_iterations", 500))
        best_w, best_fit, best_comp = genetic_algorithm_optimize(
            jm, jn, desired_consensus=desired, population_size=pop, max_iterations=iters, min_w=min_w, max_w=max_w
        )
    elif method == "HHO":
        agents = int(
            search_agents_no
            if search_agents_no is not None
            else params.get("SearchAgents_no", params.get("search_agents_no", 10))
        )
        iters = int(max_iter if max_iter is not None else params.get("Max_iter", params.get("max_iter", 100)))
        best_w, best_fit, best_comp = hho_optimize(
            jm, jn, desired_consensus=desired, search_agents_no=agents, max_iter=iters, min_w=min_w, max_w=max_w
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    best_w = _normalize_weights(best_w)
    return DHFOptimizationResult(
        dm_ids=dm_ids,
        best_weights=best_w,
        best_consensus=float(best_fit),
        compatibility=list(best_comp),
        method=method,
    )


