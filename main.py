from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from hives_dhf.dhf_consensus import optimize_expert_weights, compat3, dhf_json_to_matrices
from hives_dhf.json_input import load_decision_problem_from_json, load_decision_problem_from_data
from hives_dhf.hives_method import hives_rank

# Словарь расшифровок критериев (можно расширить при необходимости)
CRITERIA_DESCRIPTIONS = {
    "ES": "Environmental Sustainability",
    "SS": "Social Sustainability",
    "EcS": "Economic Sustainability",
    "IP": "Innovation Potential",
    "SA": "Strategic Alignment",
    "LTC": "Long-term Competitiveness",
}


def run_hives_from_json(json_path: str) -> dict:
    # Запуск расчёта по JSON: читаем данные, считаем HIVES, печатаем результаты.
    problem = load_decision_problem_from_json(json_path)

    # Матрица A (альтернатива × критерий), агрегированная по всем ЛПР
    A = problem.aggregated_performance()

    # Матрица W (эксперт × критерий)
    W = problem.weights_matrix()

    expert_ids = [e.id for e in problem.experts] if problem.experts else [f"DM{i+1}" for i in range(W.shape[0])]

    res = hives_rank(A=A, W=W, expert_ids=expert_ids)

    print("\nRaw criterion weights gamma:")
    print(np.round(res["gamma"], 2))

    print("\nScaled criterion weights gamma_scaled (sum = 100):")
    print(np.round(res["gamma_scaled"], 2))

    print("\nFinal alternative scores:")
    for idx, score in enumerate(np.round(res["alt_scores"], 2)):
        print(f"  {problem.alternatives[idx]}: {score}")

    print("\nRanking (1-based):")
    ranking_1_based = res["ranking"] + 1
    print(ranking_1_based)

    return dict(problem=problem, result=res)


def _load_json(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Hint: If you're using rotation, make sure to run the method first without rotation "
            f"to generate the required output file."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _maybe_load_predecessor_old_lambdas(
    path: str | None,
    predecessor_id: str,
    criteria_names: list[str],
) -> np.ndarray | None:
    """
    Reads previous run result JSON and extracts predecessor's lambdas row.
    Accepts:
    - lambdas_final or lambdas_base (old format)
    - lambdas.no_rotation.final/base or lambdas.with_rotation.final/base (new format with comparison)
    - lambdas.no_rotation or lambdas.with_rotation (simpler nested format)
    """
    if not path:
        return None
    data = _load_json(path)
    
    # Try new format first (lambdas.no_rotation / lambdas.with_rotation)
    lambdas_dict = data.get("lambdas")
    if lambdas_dict and isinstance(lambdas_dict, dict):
        # Check for nested structure: lambdas.no_rotation.final/base
        no_rot = lambdas_dict.get("no_rotation")
        with_rot = lambdas_dict.get("with_rotation")
        
        if no_rot and isinstance(no_rot, dict):
            # Prefer final over base
            lambdas = no_rot.get("final") or no_rot.get("base") or no_rot
        elif with_rot and isinstance(with_rot, dict):
            lambdas = with_rot.get("final") or with_rot.get("base") or with_rot
        else:
            # Simple nested: lambdas.no_rotation directly contains expert dicts
            lambdas = no_rot or with_rot
    else:
        # Fall back to old format
        lambdas = data.get("lambdas_final") or data.get("lambdas_base")
    
    if lambdas is None:
        raise ValueError(
            f"{path} must contain lambdas_final, lambdas_base, or lambdas.no_rotation/lambdas.with_rotation"
        )

    row = lambdas.get(predecessor_id)
    if row is None:
        raise ValueError(f"{path}: predecessor_id {predecessor_id!r} not found in lambdas")
    try:
        return np.array([float(row[c]) for c in criteria_names], dtype=float)
    except KeyError as e:
        raise ValueError(f"{path}: missing criterion key in predecessor lambdas: {e}") from e


def run_combined_from_json(json_path: str) -> dict:
    data = _load_json(json_path)

    hives_data = data["hives"]
    dhf_data = data["dhf"]
    rotation = data.get("rotation")
    params = data.get("combined_parameters") or {}

    dhf_method = str(params.get("dhf_method", "HHO")).upper()
    if dhf_method not in ("GA", "HHO"):
        raise ValueError("combined_parameters.dhf_method must be GA or HHO")

    influence_mode = str(params.get("influence_mode", "continuous"))
    influence_min = float(params.get("influence_min", 0.01))
    influence_max = float(params.get("influence_max", 0.99))

    output_path = params.get("output_path")

    # Load HIVES problem from embedded data
    problem = load_decision_problem_from_data(hives_data)
    A = problem.aggregated_performance()
    W = problem.weights_matrix()

    expert_ids = [e.id for e in problem.experts] if problem.experts else [f"DM{i+1}" for i in range(W.shape[0])]

    # DHF optimization
    dhf_res = optimize_expert_weights(dhf_data, method=dhf_method)  # weights sum=1

    # Ensure IDs align (as you specified: same_ids)
    if set(dhf_res.dm_ids) != set(expert_ids):
        raise ValueError(
            "IDs mismatch: DHF dms ids must match HIVES experts ids. "
            f"DHF={dhf_res.dm_ids}, HIVES={expert_ids}"
        )
    # Align influence vector to expert_ids order
    idx = {dm_id: i for i, dm_id in enumerate(dhf_res.dm_ids)}
    influence = np.array([dhf_res.best_weights[idx[eid]] for eid in expert_ids], dtype=float)

    smooth_replacement = None
    if rotation is not None:
        predecessor_id = str(rotation["predecessor_id"])
        new_id = str(rotation["new_id"])
        alpha = float(rotation["alpha"])
        prev_path = rotation.get("predecessor_old_lambdas_path")
        criteria_names = [c.name for c in problem.criteria]
        pred_old_row = _maybe_load_predecessor_old_lambdas(prev_path, predecessor_id, criteria_names)
        if pred_old_row is None:
            raise ValueError("rotation requires predecessor_old_lambdas_path to extract old lambdas")
        smooth_replacement = dict(
            predecessor_id=predecessor_id,
            new_id=new_id,
            alpha=alpha,
            predecessor_old_lambdas_row=pred_old_row,
        )

    # Always compute baseline (without smooth replacement) so we can compare.
    res_no_rotation = hives_rank(
        A=A,
        W=W,
        influence=influence,
        influence_mode=influence_mode,
        influence_min=influence_min,
        influence_max=influence_max,
        expert_ids=expert_ids,
        smooth_replacement=None,
    )

    print("\n[DHF] Best consensus:", round(dhf_res.best_consensus, 4), "method:", dhf_res.method)
    print("[DHF] Weights:", {expert_ids[i]: round(float(influence[i]), 4) for i in range(len(expert_ids))})

    def _hives_payload(res: dict) -> dict:
        return dict(
            gamma=[float(x) for x in res["gamma"]],
            gamma_scaled=[float(x) for x in res["gamma_scaled"]],
            alt_scores=[float(x) for x in res["alt_scores"]],
            ranking_1_based=[int(x) for x in (res["ranking"] + 1)],
        )

    def _lambdas_payload(res: dict) -> dict:
        return dict(
            base=_lambdas_to_dict(res.get("lambdas_base"), expert_ids, problem.criteria),
            final=_lambdas_to_dict(res.get("lambdas"), expert_ids, problem.criteria),
        )

    print("\n[HIVES][no_rotation] gamma:")
    print(np.round(res_no_rotation["gamma"], 2))
    print("\n[HIVES][no_rotation] gamma_scaled (sum = 100):")
    print(np.round(res_no_rotation["gamma_scaled"], 2))

    res_with_rotation = None
    if smooth_replacement is not None:
        res_with_rotation = hives_rank(
            A=A,
            W=W,
            influence=influence,
            influence_mode=influence_mode,
            influence_min=influence_min,
            influence_max=influence_max,
            expert_ids=expert_ids,
            smooth_replacement=smooth_replacement,
        )
        print("\n[HIVES][with_rotation] gamma:")
        print(np.round(res_with_rotation["gamma"], 2))
        print("\n[HIVES][with_rotation] gamma_scaled (sum = 100):")
        print(np.round(res_with_rotation["gamma_scaled"], 2))

    # Recompute consensus after HIVES using compat3
    print("\n" + "=" * 80)
    print("CONSENSUS COMPARISON (compat3)")
    print("=" * 80)
    
    # Convert DHF data to matrices for compat3
    jm, jn, _, _ = dhf_json_to_matrices(dhf_data)
    
    # Compute compatibilities with optimized weights (used in HIVES)
    compatibilities_after = compat3(influence, jm, jn)
    consensus_after = float(min(compatibilities_after))
    
    print("\n[BEFORE HIVES] Consensus from DHF optimization:")
    print(f"  Best consensus: {dhf_res.best_consensus:.6f}")
    print(f"  Compatibilities: {[f'{c:.6f}' for c in dhf_res.compatibility]}")
    
    print("\n[AFTER HIVES] Consensus with optimized weights (compat3):")
    print(f"  Consensus level (min): {consensus_after:.6f}")
    print(f"  Compatibilities: {[f'{c:.6f}' for c in compatibilities_after]}")
    print(f"  Expert IDs: {expert_ids}")
    
    consensus_diff = consensus_after - dhf_res.best_consensus
    print(f"\n[DIFFERENCE] After - Before: {consensus_diff:+.6f}")

    out = dict(
        dhf=dict(
            method=dhf_res.method,
            best_consensus=dhf_res.best_consensus,
            compatibility={dhf_res.dm_ids[i]: float(dhf_res.compatibility[i]) for i in range(len(dhf_res.dm_ids))}
            if dhf_res.compatibility
            else {},
            influence={expert_ids[i]: float(influence[i]) for i in range(len(expert_ids))},
        ),
        consensus_after_hives=dict(
            consensus_level=consensus_after,
            compatibilities={expert_ids[i]: float(compatibilities_after[i]) for i in range(len(expert_ids))},
            difference_from_before=float(consensus_diff),
        ),
    )

    if res_with_rotation is None:
        # Backward-compatible shape when rotation is absent
        out["hives"] = _hives_payload(res_no_rotation)
        out["lambdas_base"] = _lambdas_to_dict(res_no_rotation.get("lambdas_base"), expert_ids, problem.criteria)
        out["lambdas_final"] = _lambdas_to_dict(res_no_rotation.get("lambdas"), expert_ids, problem.criteria)
    else:
        # Comparison mode: keep both results in one output
        out["hives"] = dict(
            no_rotation=_hives_payload(res_no_rotation),
            with_rotation=_hives_payload(res_with_rotation),
        )
        out["lambdas"] = dict(
            no_rotation=_lambdas_payload(res_no_rotation),
            with_rotation=_lambdas_payload(res_with_rotation),
        )

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULT SUMMARY")
    print("=" * 80)
    
    if res_with_rotation is None:
        # Single result
        print("\n[FINAL RANKING]")
        for rank, alt_idx in enumerate(res_no_rotation["ranking"], start=1):
            alt_name = problem.alternatives[alt_idx]
            score = res_no_rotation["alt_scores"][alt_idx]
            print(f"  {rank}. {alt_name} (score: {score:.2f})")
        
        print("\n[CRITERIA WEIGHTS] (gamma_scaled, sum=100%)")
        crit_names = [c.name for c in problem.criteria]
        for i, (name, weight) in enumerate(zip(crit_names, res_no_rotation["gamma_scaled"])):
            desc = CRITERIA_DESCRIPTIONS.get(name, "")
            if desc:
                print(f"  {name} ({desc}): {weight:.2f}%")
            else:
                print(f"  {name}: {weight:.2f}%")
    else:
        # Comparison mode
        print("\n[FINAL RANKING - NO ROTATION]")
        for rank, alt_idx in enumerate(res_no_rotation["ranking"], start=1):
            alt_name = problem.alternatives[alt_idx]
            score = res_no_rotation["alt_scores"][alt_idx]
            print(f"  {rank}. {alt_name} (score: {score:.2f})")
        
        print("\n[FINAL RANKING - WITH ROTATION]")
        for rank, alt_idx in enumerate(res_with_rotation["ranking"], start=1):
            alt_name = problem.alternatives[alt_idx]
            score = res_with_rotation["alt_scores"][alt_idx]
            print(f"  {rank}. {alt_name} (score: {score:.2f})")
        
        print("\n[CRITERIA WEIGHTS - NO ROTATION] (gamma_scaled, sum=100%)")
        crit_names = [c.name for c in problem.criteria]
        for i, (name, weight) in enumerate(zip(crit_names, res_no_rotation["gamma_scaled"])):
            desc = CRITERIA_DESCRIPTIONS.get(name, "")
            if desc:
                print(f"  {name} ({desc}): {weight:.2f}%")
            else:
                print(f"  {name}: {weight:.2f}%")
        
        print("\n[CRITERIA WEIGHTS - WITH ROTATION] (gamma_scaled, sum=100%)")
        for i, (name, weight) in enumerate(zip(crit_names, res_with_rotation["gamma_scaled"])):
            desc = CRITERIA_DESCRIPTIONS.get(name, "")
            if desc:
                print(f"  {name} ({desc}): {weight:.2f}%")
            else:
                print(f"  {name}: {weight:.2f}%")

    if output_path:
        Path(output_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved combined result to: {output_path}")

    return out


def _lambdas_to_dict(lambdas: np.ndarray | None, expert_ids: list[str], criteria: list) -> dict:
    if lambdas is None:
        return {}
    lambdas = np.asarray(lambdas, dtype=float)
    crit_names = [c.name for c in criteria]
    out: dict = {}
    for i, eid in enumerate(expert_ids):
        out[eid] = {crit_names[j]: float(lambdas[i, j]) for j in range(len(crit_names))}
    return out


def run_experiment(json_path: str) -> dict:
    """
    Эксперимент: сравнение весов экспертов в разных сценариях:
    1. HIVES без DHF (равномерные веса)
    2. HIVES + DHF (GA) без замены
    3. HIVES + DHF (GA) с заменой
    4. Только DHF (GA) - веса экспертов
    """
    data = _load_json(json_path)
    hives_data = data["hives"]
    dhf_data = data["dhf"]
    rotation = data.get("rotation")
    params = data.get("combined_parameters") or {}

    # Load problem
    problem = load_decision_problem_from_data(hives_data)
    A = problem.aggregated_performance()
    W = problem.weights_matrix()
    expert_ids = [e.id for e in problem.experts] if problem.experts else [f"DM{i+1}" for i in range(W.shape[0])]

    results = {}

    print("=" * 80)
    print("EXPERIMENT: Expert Weights Comparison")
    print("=" * 80)

    # 1. HIVES без DHF (равномерные веса)
    print("\n[1] HIVES without DHF (uniform expert weights)...")
    uniform_influence = np.ones(len(expert_ids), dtype=float) / len(expert_ids)
    res_hives_only = hives_rank(A=A, W=W, influence=uniform_influence, expert_ids=expert_ids)
    results["hives_only"] = {
        "expert_weights": {expert_ids[i]: float(uniform_influence[i]) for i in range(len(expert_ids))},
        "lambdas": _lambdas_to_dict(res_hives_only.get("lambdas"), expert_ids, problem.criteria),
    }

    # 2. Только DHF (GA) - веса экспертов
    print("\n[2] DHF (GA) only - expert weights optimization...")
    dhf_res_ga = optimize_expert_weights(dhf_data, method="GA")
    idx_ga = {dm_id: i for i, dm_id in enumerate(dhf_res_ga.dm_ids)}
    influence_ga = np.array([dhf_res_ga.best_weights[idx_ga[eid]] for eid in expert_ids], dtype=float)
    results["dhf_ga_only"] = {
        "expert_weights": {expert_ids[i]: float(influence_ga[i]) for i in range(len(expert_ids))},
        "consensus": float(dhf_res_ga.best_consensus),
    }

    # 3. HIVES + DHF (GA) без замены
    print("\n[3] HIVES + DHF (GA) without rotation...")
    res_hives_ga_no_rot = hives_rank(
        A=A,
        W=W,
        influence=influence_ga,
        influence_mode=str(params.get("influence_mode", "continuous")),
        influence_min=float(params.get("influence_min", 0.01)),
        influence_max=float(params.get("influence_max", 0.99)),
        expert_ids=expert_ids,
    )
    results["hives_ga_no_rotation"] = {
        "expert_weights": {expert_ids[i]: float(influence_ga[i]) for i in range(len(expert_ids))},
        "lambdas": _lambdas_to_dict(res_hives_ga_no_rot.get("lambdas"), expert_ids, problem.criteria),
    }

    # 4. HIVES + DHF (GA) с заменой (если rotation задан)
    if rotation is not None:
        print("\n[4] HIVES + DHF (GA) with rotation...")
        predecessor_id = str(rotation["predecessor_id"])
        new_id = str(rotation["new_id"])
        alpha = float(rotation["alpha"])
        prev_path = rotation.get("predecessor_old_lambdas_path")
        criteria_names = [c.name for c in problem.criteria]
        pred_old_row = _maybe_load_predecessor_old_lambdas(prev_path, predecessor_id, criteria_names)
        if pred_old_row is None:
            print(f"  WARNING: Cannot load predecessor lambdas from {prev_path}, skipping rotation scenario")
            results["hives_ga_with_rotation"] = None
        else:
            smooth_replacement = dict(
                predecessor_id=predecessor_id,
                new_id=new_id,
                alpha=alpha,
                predecessor_old_lambdas_row=pred_old_row,
            )
            res_hives_ga_with_rot = hives_rank(
                A=A,
                W=W,
                influence=influence_ga,
                influence_mode=str(params.get("influence_mode", "continuous")),
                influence_min=float(params.get("influence_min", 0.01)),
                influence_max=float(params.get("influence_max", 0.99)),
                expert_ids=expert_ids,
                smooth_replacement=smooth_replacement,
            )
            results["hives_ga_with_rotation"] = {
                "expert_weights": {expert_ids[i]: float(influence_ga[i]) for i in range(len(expert_ids))},
                "lambdas": _lambdas_to_dict(res_hives_ga_with_rot.get("lambdas"), expert_ids, problem.criteria),
                "rotation_params": {"predecessor_id": predecessor_id, "new_id": new_id, "alpha": alpha},
            }
    else:
        print("\n[4] HIVES + DHF (GA) with rotation: SKIPPED (no rotation block in JSON)")
        results["hives_ga_with_rotation"] = None

    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON: Expert Weights")
    print("=" * 80)
    print(f"\n{'Expert':<10} {'HIVES only':<15} {'DHF(GA) only':<15} {'HIVES+GA (no rot)':<18} {'HIVES+GA (rot)':<18}")
    print("-" * 80)
    for eid in expert_ids:
        hives_only_w = results["hives_only"]["expert_weights"].get(eid, 0.0)
        dhf_ga_w = results["dhf_ga_only"]["expert_weights"].get(eid, 0.0)
        hives_ga_no_w = results["hives_ga_no_rotation"]["expert_weights"].get(eid, 0.0)
        hives_ga_rot_w = (
            results["hives_ga_with_rotation"]["expert_weights"].get(eid, 0.0)
            if results["hives_ga_with_rotation"] is not None
            else None
        )
        rot_str = f"{hives_ga_rot_w:.4f}" if hives_ga_rot_w is not None else "N/A"
        print(f"{eid:<10} {hives_only_w:<15.4f} {dhf_ga_w:<15.4f} {hives_ga_no_w:<18.4f} {rot_str:<18}")

    # Print lambda comparison (average across criteria)
    print("\n" + "=" * 80)
    print("COMPARISON: Average Lambda (across all criteria)")
    print("=" * 80)
    print(f"\n{'Expert':<10} {'HIVES only':<15} {'HIVES+GA (no rot)':<18} {'HIVES+GA (rot)':<18}")
    print("-" * 80)
    for eid in expert_ids:
        lambdas_hives = results["hives_only"]["lambdas"].get(eid, {})
        lambdas_ga_no = results["hives_ga_no_rotation"]["lambdas"].get(eid, {})
        lambdas_ga_rot = (
            results["hives_ga_with_rotation"]["lambdas"].get(eid, {})
            if results["hives_ga_with_rotation"] is not None
            else {}
        )
        avg_hives = np.mean(list(lambdas_hives.values())) if lambdas_hives else 0.0
        avg_ga_no = np.mean(list(lambdas_ga_no.values())) if lambdas_ga_no else 0.0
        avg_ga_rot = np.mean(list(lambdas_ga_rot.values())) if lambdas_ga_rot else 0.0
        rot_str = f"{avg_ga_rot:.2f}" if lambdas_ga_rot else "N/A"
        print(f"{eid:<10} {avg_hives:<15.2f} {avg_ga_no:<18.2f} {rot_str:<18}")

    # Save results
    output_path = params.get("output_path", "outputs/experiment_result.json")
    output_path = output_path.replace(".json", "_experiment.json") if output_path.endswith(".json") else output_path
    Path(output_path).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved experiment results to: {output_path}")

    return results


def compare_order(json_path: str) -> dict:
    """
    Сравнение: что будет, если поменять местами DHF и HIVES?
    Показывает разницу между:
    1. DHF -> HIVES (обычный порядок)
    2. HIVES -> DHF (обратный порядок)
    """
    data = _load_json(json_path)
    hives_data = data["hives"]
    dhf_data = data["dhf"]
    params = data.get("combined_parameters") or {}

    dhf_method = str(params.get("dhf_method", "HHO")).upper()
    influence_mode = str(params.get("influence_mode", "continuous"))
    influence_min = float(params.get("influence_min", 0.01))
    influence_max = float(params.get("influence_max", 0.99))

    problem = load_decision_problem_from_data(hives_data)
    A = problem.aggregated_performance()
    W = problem.weights_matrix()
    expert_ids = [e.id for e in problem.experts] if problem.experts else [f"DM{i+1}" for i in range(W.shape[0])]

    print("=" * 80)
    print("COMPARISON: DHF->HIVES vs HIVES->DHF")
    print("=" * 80)

    # ===== ПОДХОД 1: DHF -> HIVES (обычный порядок) =====
    print("\n[APPROACH 1] DHF -> HIVES (normal order)")
    print("-" * 80)
    
    dhf_res = optimize_expert_weights(dhf_data, method=dhf_method)
    idx = {dm_id: i for i, dm_id in enumerate(dhf_res.dm_ids)}
    influence_dhf = np.array([dhf_res.best_weights[idx[eid]] for eid in expert_ids], dtype=float)
    
    print(f"DHF optimized weights: {[f'{w:.4f}' for w in influence_dhf]}")
    print(f"DHF consensus: {dhf_res.best_consensus:.6f}")
    
    res_hives_after_dhf = hives_rank(
        A=A, W=W, influence=influence_dhf,
        influence_mode=influence_mode, influence_min=influence_min, influence_max=influence_max,
        expert_ids=expert_ids,
    )
    
    print(f"HIVES ranking (with DHF weights): {[problem.alternatives[i] for i in res_hives_after_dhf['ranking']]}")
    print(f"HIVES scores: {[f'{s:.2f}' for s in res_hives_after_dhf['alt_scores']]}")

    # ===== ПОДХОД 2: HIVES -> DHF (обратный порядок) =====
    print("\n[APPROACH 2] HIVES -> DHF (reversed order)")
    print("-" * 80)
    
    # HIVES с равномерными весами (так как DHF еще не выполнен)
    uniform_influence = np.ones(len(expert_ids), dtype=float) / len(expert_ids)
    print(f"Using uniform weights for HIVES: {[f'{w:.4f}' for w in uniform_influence]}")
    
    res_hives_before_dhf = hives_rank(
        A=A, W=W, influence=uniform_influence,
        influence_mode=influence_mode, influence_min=influence_min, influence_max=influence_max,
        expert_ids=expert_ids,
    )
    
    print(f"HIVES ranking (with uniform weights): {[problem.alternatives[i] for i in res_hives_before_dhf['ranking']]}")
    print(f"HIVES scores: {[f'{s:.2f}' for s in res_hives_before_dhf['alt_scores']]}")
    
    # Теперь DHF (но его результаты уже не влияют на HIVES)
    dhf_res_after_hives = optimize_expert_weights(dhf_data, method=dhf_method)
    idx2 = {dm_id: i for i, dm_id in enumerate(dhf_res_after_hives.dm_ids)}
    influence_dhf_after = np.array([dhf_res_after_hives.best_weights[idx2[eid]] for eid in expert_ids], dtype=float)
    
    print(f"DHF optimized weights (after HIVES): {[f'{w:.4f}' for w in influence_dhf_after]}")
    print(f"DHF consensus: {dhf_res_after_hives.best_consensus:.6f}")
    print("NOTE: These DHF weights are NOT used in HIVES (HIVES already executed)")

    # ===== СРАВНЕНИЕ =====
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    
    print("\n[Ranking Comparison]")
    ranking1 = [problem.alternatives[i] for i in res_hives_after_dhf['ranking']]
    ranking2 = [problem.alternatives[i] for i in res_hives_before_dhf['ranking']]
    print(f"  DHF->HIVES: {ranking1}")
    print(f"  HIVES->DHF: {ranking2}")
    print(f"  Same ranking? {ranking1 == ranking2}")
    
    print("\n[Expert Weights Comparison]")
    print(f"  DHF->HIVES weights: {[f'{w:.4f}' for w in influence_dhf]}")
    print(f"  HIVES->DHF weights: {[f'{w:.4f}' for w in influence_dhf_after]}")
    print(f"  Same weights? {np.allclose(influence_dhf, influence_dhf_after, atol=1e-6)}")
    
    print("\n[Key Difference]")
    print("  DHF->HIVES: DHF weights ARE used in HIVES -> integrated result")
    print("  HIVES->DHF: DHF weights are NOT used in HIVES -> independent results")

    return {
        "dhf_then_hives": {
            "dhf_weights": [float(w) for w in influence_dhf],
            "dhf_consensus": float(dhf_res.best_consensus),
            "hives_ranking": [problem.alternatives[i] for i in res_hives_after_dhf['ranking']],
            "hives_scores": [float(s) for s in res_hives_after_dhf['alt_scores']],
        },
        "hives_then_dhf": {
            "hives_ranking": [problem.alternatives[i] for i in res_hives_before_dhf['ranking']],
            "hives_scores": [float(s) for s in res_hives_before_dhf['alt_scores']],
            "dhf_weights": [float(w) for w in influence_dhf_after],
            "dhf_consensus": float(dhf_res_after_hives.best_consensus),
        },
    }


def run_hives_compat3_dhf(json_path: str) -> dict:
    """
    Новый порядок выполнения: HIVES -> compat3 -> DHF
    1. Сначала выполняется HIVES (с равномерными весами)
    2. Затем compat3 (для проверки консенсуса с равномерными весами)
    3. Затем DHF (оптимизация весов экспертов)
    """
    data = _load_json(json_path)
    hives_data = data["hives"]
    dhf_data = data["dhf"]
    rotation = data.get("rotation")
    params = data.get("combined_parameters") or {}

    dhf_method = str(params.get("dhf_method", "HHO")).upper()
    if dhf_method not in ("GA", "HHO"):
        raise ValueError("combined_parameters.dhf_method must be GA or HHO")

    influence_mode = str(params.get("influence_mode", "continuous"))
    influence_min = float(params.get("influence_min", 0.01))
    influence_max = float(params.get("influence_max", 0.99))

    output_path = params.get("output_path")

    # Load HIVES problem
    problem = load_decision_problem_from_data(hives_data)
    A = problem.aggregated_performance()
    W = problem.weights_matrix()
    expert_ids = [e.id for e in problem.experts] if problem.experts else [f"DM{i+1}" for i in range(W.shape[0])]

    print("=" * 80)
    print("EXECUTION ORDER: HIVES -> compat3 -> DHF")
    print("=" * 80)

    # ===== ШАГ 1: HIVES (с равномерными весами) =====
    print("\n[STEP 1] HIVES (with uniform expert weights)...")
    uniform_influence = np.ones(len(expert_ids), dtype=float) / len(expert_ids)
    print(f"  Using uniform weights: {[f'{w:.4f}' for w in uniform_influence]}")

    smooth_replacement = None
    if rotation is not None:
        predecessor_id = str(rotation["predecessor_id"])
        new_id = str(rotation["new_id"])
        alpha = float(rotation["alpha"])
        prev_path = rotation.get("predecessor_old_lambdas_path")
        criteria_names = [c.name for c in problem.criteria]
        pred_old_row = _maybe_load_predecessor_old_lambdas(prev_path, predecessor_id, criteria_names)
        if pred_old_row is None:
            raise ValueError("rotation requires predecessor_old_lambdas_path to extract old lambdas")
        smooth_replacement = dict(
            predecessor_id=predecessor_id,
            new_id=new_id,
            alpha=alpha,
            predecessor_old_lambdas_row=pred_old_row,
        )

    res_no_rotation = hives_rank(
        A=A, W=W, influence=uniform_influence,
        influence_mode=influence_mode, influence_min=influence_min, influence_max=influence_max,
        expert_ids=expert_ids, smooth_replacement=None,
    )

    print("\n[HIVES] Raw criterion weights gamma:")
    print(np.round(res_no_rotation["gamma"], 2))
    print("\n[HIVES] Scaled criterion weights gamma_scaled (sum = 100):")
    print(np.round(res_no_rotation["gamma_scaled"], 2))
    print("\n[HIVES] Final alternative scores:")
    for idx, score in enumerate(np.round(res_no_rotation["alt_scores"], 2)):
        print(f"  {problem.alternatives[idx]}: {score}")
    print("\n[HIVES] Ranking (1-based):")
    ranking_1_based = res_no_rotation["ranking"] + 1
    print(ranking_1_based)

    res_with_rotation = None
    if smooth_replacement is not None:
        res_with_rotation = hives_rank(
            A=A, W=W, influence=uniform_influence,
            influence_mode=influence_mode, influence_min=influence_min, influence_max=influence_max,
            expert_ids=expert_ids, smooth_replacement=smooth_replacement,
        )
        print("\n[HIVES][with_rotation] gamma:")
        print(np.round(res_with_rotation["gamma"], 2))
        print("\n[HIVES][with_rotation] gamma_scaled (sum = 100):")
        print(np.round(res_with_rotation["gamma_scaled"], 2))

    # ===== ШАГ 2: compat3 (с равномерными весами) =====
    print("\n" + "=" * 80)
    print("[STEP 2] compat3 (with uniform weights)")
    print("=" * 80)

    jm, jn, _, _ = dhf_json_to_matrices(dhf_data)
    compatibilities_before_dhf = compat3(uniform_influence, jm, jn)
    consensus_before_dhf = float(min(compatibilities_before_dhf))

    print(f"\nConsensus level (min): {consensus_before_dhf:.6f}")
    print(f"Compatibilities: {[f'{c:.6f}' for c in compatibilities_before_dhf]}")
    print(f"Expert IDs: {expert_ids}")

    # ===== ШАГ 3: DHF (оптимизация весов) =====
    print("\n" + "=" * 80)
    print("[STEP 3] DHF (optimization of expert weights)")
    print("=" * 80)

    dhf_res = optimize_expert_weights(dhf_data, method=dhf_method)
    idx = {dm_id: i for i, dm_id in enumerate(dhf_res.dm_ids)}
    influence_optimized = np.array([dhf_res.best_weights[idx[eid]] for eid in expert_ids], dtype=float)

    print(f"\nOptimized expert weights: {[f'{w:.4f}' for w in influence_optimized]}")
    print(f"Best consensus: {dhf_res.best_consensus:.6f}")
    print(f"Compatibilities: {[f'{c:.6f}' for c in dhf_res.compatibility]}")

    # Пересчитываем compat3 с оптимизированными весами
    compatibilities_after_dhf = compat3(influence_optimized, jm, jn)
    consensus_after_dhf = float(min(compatibilities_after_dhf))

    print(f"\n[compat3 with optimized weights]")
    print(f"  Consensus level (min): {consensus_after_dhf:.6f}")
    print(f"  Compatibilities: {[f'{c:.6f}' for c in compatibilities_after_dhf]}")

    # Сравнение
    print("\n" + "=" * 80)
    print("COMPARISON: Before vs After DHF optimization")
    print("=" * 80)
    print(f"\nConsensus (uniform weights): {consensus_before_dhf:.6f}")
    print(f"Consensus (optimized weights): {consensus_after_dhf:.6f}")
    print(f"Improvement: {consensus_after_dhf - consensus_before_dhf:+.6f}")

    # Формируем результат
    def _hives_payload(res: dict) -> dict:
        return dict(
            gamma=[float(x) for x in res["gamma"]],
            gamma_scaled=[float(x) for x in res["gamma_scaled"]],
            alt_scores=[float(x) for x in res["alt_scores"]],
            ranking_1_based=[int(x) for x in (res["ranking"] + 1)],
        )

    def _lambdas_payload(res: dict) -> dict:
        return dict(
            base=_lambdas_to_dict(res.get("lambdas_base"), expert_ids, problem.criteria),
            final=_lambdas_to_dict(res.get("lambdas"), expert_ids, problem.criteria),
        )

    out = dict(
        execution_order="HIVES -> compat3 -> DHF",
        hives=dict(
            expert_weights_uniform={expert_ids[i]: float(uniform_influence[i]) for i in range(len(expert_ids))},
        ),
        compat3_before_dhf=dict(
            consensus_level=consensus_before_dhf,
            compatibilities={expert_ids[i]: float(compatibilities_before_dhf[i]) for i in range(len(expert_ids))},
        ),
        dhf=dict(
            method=dhf_res.method,
            best_consensus=dhf_res.best_consensus,
            compatibility={dhf_res.dm_ids[i]: float(dhf_res.compatibility[i]) for i in range(len(dhf_res.dm_ids))}
            if dhf_res.compatibility
            else {},
            influence={expert_ids[i]: float(influence_optimized[i]) for i in range(len(expert_ids))},
        ),
        compat3_after_dhf=dict(
            consensus_level=consensus_after_dhf,
            compatibilities={expert_ids[i]: float(compatibilities_after_dhf[i]) for i in range(len(expert_ids))},
            improvement=float(consensus_after_dhf - consensus_before_dhf),
        ),
    )

    if res_with_rotation is None:
        out["hives"].update(_hives_payload(res_no_rotation))
        out["lambdas_base"] = _lambdas_to_dict(res_no_rotation.get("lambdas_base"), expert_ids, problem.criteria)
        out["lambdas_final"] = _lambdas_to_dict(res_no_rotation.get("lambdas"), expert_ids, problem.criteria)
    else:
        out["hives"].update(dict(
            no_rotation=_hives_payload(res_no_rotation),
            with_rotation=_hives_payload(res_with_rotation),
        ))
        out["lambdas"] = dict(
            no_rotation=_lambdas_payload(res_no_rotation),
            with_rotation=_lambdas_payload(res_with_rotation),
        )

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULT SUMMARY")
    print("=" * 80)

    if res_with_rotation is None:
        print("\n[FINAL RANKING]")
        for rank, alt_idx in enumerate(res_no_rotation["ranking"], start=1):
            alt_name = problem.alternatives[alt_idx]
            score = res_no_rotation["alt_scores"][alt_idx]
            print(f"  {rank}. {alt_name} (score: {score:.2f})")

        print("\n[CRITERIA WEIGHTS] (gamma_scaled, sum=100%)")
        crit_names = [c.name for c in problem.criteria]
        for i, (name, weight) in enumerate(zip(crit_names, res_no_rotation["gamma_scaled"])):
            desc = CRITERIA_DESCRIPTIONS.get(name, "")
            if desc:
                print(f"  {name} ({desc}): {weight:.2f}%")
            else:
                print(f"  {name}: {weight:.2f}%")
    else:
        print("\n[FINAL RANKING - NO ROTATION]")
        for rank, alt_idx in enumerate(res_no_rotation["ranking"], start=1):
            alt_name = problem.alternatives[alt_idx]
            score = res_no_rotation["alt_scores"][alt_idx]
            print(f"  {rank}. {alt_name} (score: {score:.2f})")

        print("\n[FINAL RANKING - WITH ROTATION]")
        for rank, alt_idx in enumerate(res_with_rotation["ranking"], start=1):
            alt_name = problem.alternatives[alt_idx]
            score = res_with_rotation["alt_scores"][alt_idx]
            print(f"  {rank}. {alt_name} (score: {score:.2f})")

        print("\n[CRITERIA WEIGHTS - NO ROTATION] (gamma_scaled, sum=100%)")
        crit_names = [c.name for c in problem.criteria]
        for i, (name, weight) in enumerate(zip(crit_names, res_no_rotation["gamma_scaled"])):
            desc = CRITERIA_DESCRIPTIONS.get(name, "")
            if desc:
                print(f"  {name} ({desc}): {weight:.2f}%")
            else:
                print(f"  {name}: {weight:.2f}%")

        print("\n[CRITERIA WEIGHTS - WITH ROTATION] (gamma_scaled, sum=100%)")
        for i, (name, weight) in enumerate(zip(crit_names, res_with_rotation["gamma_scaled"])):
            desc = CRITERIA_DESCRIPTIONS.get(name, "")
            if desc:
                print(f"  {name} ({desc}): {weight:.2f}%")
            else:
                print(f"  {name}: {weight:.2f}%")

    if output_path:
        Path(output_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved result to: {output_path}")

    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HIVES и комбинированный HIVES+DHF пайплайн (JSON).")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p_hives = sub.add_parser("hives", help="Запуск HIVES по input.json (как раньше).")
    p_hives.add_argument("json_path", type=str, help="Путь к JSON-файлу для HIVES.")
    p_hives.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Опционально: сохранить результат HIVES в JSON.",
    )

    p_comb = sub.add_parser("combined", help="Запуск DHF->(influence)->HIVES по combined JSON.")
    p_comb.add_argument("json_path", type=str, help="Путь к combined JSON.")

    p_exp = sub.add_parser(
        "experiment",
        help="Эксперимент: сравнение весов экспертов (HIVES, HIVES+GA, GA only, с/без замены).",
    )
    p_exp.add_argument("json_path", type=str, help="Путь к combined JSON.")

    p_compare = sub.add_parser(
        "compare-order",
        help="Сравнение: что будет, если поменять местами DHF и HIVES?",
    )
    p_compare.add_argument("json_path", type=str, help="Путь к combined JSON.")

    p_hives_compat_dhf = sub.add_parser(
        "hives-compat-dhf",
        help="Новый порядок: HIVES -> compat3 -> DHF",
    )
    p_hives_compat_dhf.add_argument("json_path", type=str, help="Путь к combined JSON.")

    return parser.parse_args()


if __name__ == "__main__":
    # Backward compatible mode: `python main.py input.json`
    # We do this before argparse to avoid conflicts between positional subcommands and positional json_path.
    if len(sys.argv) == 2 and sys.argv[1].lower().endswith(".json"):
        run_hives_from_json(sys.argv[1])
        raise SystemExit(0)

    args = parse_args()
    if args.cmd == "combined":
        run_combined_from_json(args.json_path)
    elif args.cmd == "experiment":
        run_experiment(args.json_path)
    elif args.cmd == "compare-order":
        compare_order(args.json_path)
    elif args.cmd == "hives-compat-dhf":
        run_hives_compat3_dhf(args.json_path)
    elif args.cmd == "hives":
        out = run_hives_from_json(args.json_path)
        if args.output:
            problem = out["problem"]
            res = out["result"]
            expert_ids = [e.id for e in problem.experts] if problem.experts else [f"DM{i+1}" for i in range(problem.weights_matrix().shape[0])]
            payload = dict(
                hives=dict(
                    gamma=[float(x) for x in res["gamma"]],
                    gamma_scaled=[float(x) for x in res["gamma_scaled"]],
                    alt_scores=[float(x) for x in res["alt_scores"]],
                    ranking_1_based=[int(x) for x in (res["ranking"] + 1)],
                ),
                lambdas_base=_lambdas_to_dict(res.get("lambdas_base"), expert_ids, problem.criteria),
                lambdas_final=_lambdas_to_dict(res.get("lambdas"), expert_ids, problem.criteria),
            )
            Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nSaved HIVES result to: {args.output}")
    else:
        raise SystemExit(
            "Usage: python main.py hives input.json  OR  python main.py combined combined.json  OR  python main.py experiment combined.json  OR  python main.py compare-order combined.json  OR  python main.py hives-compat-dhf combined.json"
        )

