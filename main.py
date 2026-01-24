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

    def _key_variants(k: str) -> list[str]:
        # Allow some common renamings between runs (e.g. "Criterion_1" vs "Criterion 1")
        variants = [k]
        if "_" in k:
            variants.append(k.replace("_", " "))
        if k.startswith("Criterion_"):
            variants.append("Criterion " + k.split("_", 1)[1])
        return list(dict.fromkeys(variants))

    out = []
    missing: list[str] = []
    for c in criteria_names:
        found = False
        for kk in _key_variants(c):
            if kk in row:
                out.append(float(row[kk]))
                found = True
                break
        if not found:
            missing.append(c)

    if missing:
        raise ValueError(
            f"{path}: predecessor lambdas keys do not match current criteria names. "
            f"Missing (first 5): {missing[:5]} (total {len(missing)})."
        )
    return np.array(out, dtype=float)


def _maybe_load_initial_influence_from_result(
    path: str | None,
    expert_ids: list[str],
) -> np.ndarray | None:
    """
    Reads previous run result JSON and extracts expert influence weights vector aligned to expert_ids.
    Accepts (best-effort):
      - out["dhf"]["influence"] as {id: weight}
      - out["influence"] as {id: weight}
    Returns None if not found.
    """
    if not path:
        return None
    data = _load_json(path)
    influence = None
    if isinstance(data.get("dhf"), dict) and isinstance(data["dhf"].get("influence"), dict):
        influence = data["dhf"]["influence"]
    elif isinstance(data.get("influence"), dict):
        influence = data["influence"]
    if not isinstance(influence, dict):
        return None

    w = []
    for eid in expert_ids:
        if eid not in influence:
            return None
        w.append(float(influence[eid]))
    w = np.asarray(w, dtype=float)
    s = float(w.sum())
    if s <= 0:
        return None
    return w / s


def _filter_dms_by_ids(dms: list[dict], keep_ids: set[str]) -> list[dict]:
    return [dm for dm in dms if str(dm.get("id")) in keep_ids]


def _build_step2_combined_input(
    combined: dict,
    predecessor_id: str,
    keep_ids: list[str],
) -> dict:
    """
    Creates a "step2" combined input JSON:
    - removes predecessor_id from dhf.dms and hives.experts
    - keeps only keep_ids (order preserved as in keep_ids)
    - leaves hives.dms untouched unless it looks like it matches experts ids (then filters too)
    """
    keep_set = set(keep_ids)
    if predecessor_id in keep_set:
        raise ValueError("keep_ids must not contain predecessor_id")

    hives = dict(combined["hives"])
    dhf = dict(combined["dhf"])

    # ---- DHF side: always filter dms ----
    dhf_dms = list(dhf.get("dms") or [])
    dhf["dms"] = _filter_dms_by_ids(dhf_dms, keep_set)

    # ---- HIVES side: filter experts if present ----
    if "experts" in hives and hives["experts"] is not None:
        experts = list(hives["experts"])
        experts_by_id = {str(e["id"]): e for e in experts}
        hives["experts"] = [experts_by_id[eid] for eid in keep_ids if eid in experts_by_id]

    # Optionally filter hives.dms if they match expert IDs (common "DM==expert" case).
    if "dms" in hives and isinstance(hives["dms"], list) and hives["dms"]:
        dm_ids = [str(x.get("id")) for x in hives["dms"]]
        # Heuristic: filter only if there is a meaningful overlap (avoid deleting sole DM used for scores).
        if any(did in keep_set or did == predecessor_id for did in dm_ids) and len(hives["dms"]) > 1:
            hives["dms"] = _filter_dms_by_ids(list(hives["dms"]), keep_set)

    out = dict(combined)
    out["hives"] = hives
    out["dhf"] = dhf
    return out


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


def run_pipeline_replace_ga_hho(json_path: str) -> dict:
    """
    End-to-end pipeline matching the requested structure:
      - Load combined input (can contain 5 experts: 4 active + 1 reserve)
      - Build step2 combined input with 4 experts by removing predecessor_id
      - Check consensus via compat3 (baseline + optional initial weights)
      - Run GA and HHO separately (optionally seeded with initial weights)
      - Run HIVES with GA weights and with HHO weights (and apply smooth replacement)
      - Save: step2 input, HIVES+GA result, HIVES+HHO result
    """
    data = _load_json(json_path)
    if "rotation" not in data:
        raise ValueError("pipeline requires 'rotation' block (predecessor_id, new_id, alpha, predecessor_old_lambdas_path)")

    rotation = data["rotation"]
    predecessor_id = str(rotation["predecessor_id"])
    new_id = str(rotation["new_id"])
    alpha = float(rotation["alpha"])
    prev_path = rotation.get("predecessor_old_lambdas_path")

    params = data.get("combined_parameters") or {}
    influence_mode = str(params.get("influence_mode", "continuous"))
    influence_min = float(params.get("influence_min", 0.01))
    influence_max = float(params.get("influence_max", 0.99))

    # Output paths
    step2_path = str(params.get("step2_output_path", "outputs/step2_combined.json"))
    out_ga_path = str(params.get("output_path_ga", "outputs/result_hives_ga.json"))
    out_hho_path = str(params.get("output_path_hho", "outputs/result_hives_hho.json"))

    # Expert IDs in input (prefer HIVES experts, else DHF DMs)
    hives_data = data["hives"]
    dhf_data = data["dhf"]
    if "experts" in hives_data and hives_data["experts"]:
        input_expert_ids = [str(e["id"]) for e in hives_data["experts"]]
    else:
        input_expert_ids = [str(dm["id"]) for dm in (dhf_data.get("dms") or [])]

    if predecessor_id not in input_expert_ids:
        raise ValueError(f"predecessor_id {predecessor_id!r} must be present among experts in the input")
    if new_id not in input_expert_ids:
        raise ValueError(f"new_id {new_id!r} must be present among experts in the input")

    # Build "step2" experts = all except predecessor_id (keeps reserve/new_id)
    keep_ids = [eid for eid in input_expert_ids if eid != predecessor_id]
    if new_id not in keep_ids:
        raise ValueError("Internal error: new_id must be kept in step2")

    step2_combined = _build_step2_combined_input(data, predecessor_id=predecessor_id, keep_ids=keep_ids)
    Path(step2_path).write_text(json.dumps(step2_combined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved step2 combined input (4 experts) to: {step2_path}")

    # Prepare HIVES problem from step2
    problem = load_decision_problem_from_data(step2_combined["hives"])
    A = problem.aggregated_performance()
    W = problem.weights_matrix()
    expert_ids = [e.id for e in problem.experts] if problem.experts else [f"DM{i+1}" for i in range(W.shape[0])]

    # Ensure DHF ids align with step2 experts
    jm, jn, _crit, dm_ids = dhf_json_to_matrices(step2_combined["dhf"])
    if set(dm_ids) != set(expert_ids):
        raise ValueError(
            "IDs mismatch in step2: DHF dms ids must match HIVES experts ids. "
            f"DHF={dm_ids}, HIVES={expert_ids}"
        )
    # Align DHF matrices DM order to expert_ids order if needed
    if dm_ids != expert_ids:
        idx = {dm_id: i for i, dm_id in enumerate(dm_ids)}
        order = [idx[eid] for eid in expert_ids]
        jm = jm[:, :, :, order]
        jn = jn[:, :, :, order]
        dm_ids = expert_ids

    # Load predecessor old lambdas row from previous results.
    # If the referenced file is incompatible (different criteria naming/size), fall back to
    # computing predecessor lambdas from the original (pre-step2) group contained in the input.
    criteria_names = [c.name for c in problem.criteria]
    pred_old_row = None
    try:
        pred_old_row = _maybe_load_predecessor_old_lambdas(prev_path, predecessor_id, criteria_names)
    except Exception as e:
        print(f"[pipeline] WARNING: cannot load predecessor lambdas from {prev_path!r}: {e}")
        print("[pipeline]          Falling back to computing predecessor lambdas from the input group (before step2).")

    if pred_old_row is None:
        # Compute from the original group in input JSON (before removing predecessor_id)
        original_problem = load_decision_problem_from_data(data["hives"])
        W0 = original_problem.weights_matrix()
        expert_ids0 = (
            [e.id for e in original_problem.experts]
            if original_problem.experts
            else [f"DM{i+1}" for i in range(W0.shape[0])]
        )
        if predecessor_id not in expert_ids0:
            raise ValueError(
                "pipeline fallback failed: predecessor_id is not present in input hives.experts "
                f"(predecessor_id={predecessor_id!r})"
            )
        # A is not needed for lambdas, but hives_rank expects it; reuse aggregated performance.
        A0 = original_problem.aggregated_performance()
        base_res = hives_rank(A=A0, W=W0, influence=None, expert_ids=expert_ids0)
        pred_idx = expert_ids0.index(predecessor_id)
        pred_old_row = np.asarray(base_res["lambdas_base"][pred_idx, :], dtype=float)
        if pred_old_row.shape[0] != len(criteria_names):
            raise ValueError(
                "pipeline fallback failed: predecessor lambdas length does not match step2 criteria count "
                f"({pred_old_row.shape[0]} vs {len(criteria_names)})"
            )

    smooth_replacement = dict(
        predecessor_id=predecessor_id,
        new_id=new_id,
        alpha=alpha,
        predecessor_old_lambdas_row=pred_old_row,
    )

    # Optional: initial (non-equal) weights from previous result file, restricted to step2 experts
    initial_influence_full = _maybe_load_initial_influence_from_result(prev_path, input_expert_ids)
    initial_influence = None
    if initial_influence_full is not None:
        idx_full = {eid: i for i, eid in enumerate(input_expert_ids)}
        w = np.array([initial_influence_full[idx_full[eid]] for eid in expert_ids], dtype=float)
        s = float(w.sum())
        if s > 0:
            initial_influence = w / s

    # Step 6: consensus check (compat3)
    uniform = np.ones(len(expert_ids), dtype=float) / len(expert_ids)
    compat_uniform = compat3(uniform, jm, jn)
    consensus_uniform = float(min(compat_uniform))
    print(f"[STEP6] Consensus (uniform weights): {consensus_uniform:.6f}")

    consensus_initial = None
    if initial_influence is not None:
        compat_init = compat3(initial_influence, jm, jn)
        consensus_initial = float(min(compat_init))
        print(f"[STEP6] Consensus (initial weights from prev result): {consensus_initial:.6f}")

    # Step 7+8+9: GA and HHO separately, then HIVES
    def _run_one(method: str, out_path: str) -> dict:
        dhf_res = optimize_expert_weights(
            step2_combined["dhf"],
            method=method,  # GA or HHO
            initial_weights=initial_influence,
        )
        influence = np.array([dhf_res.best_weights[dm_ids.index(eid)] for eid in expert_ids], dtype=float)
        compat_after = compat3(influence, jm, jn)
        consensus_after = float(min(compat_after))

        res_hives = hives_rank(
            A=A,
            W=W,
            influence=influence,
            influence_mode=influence_mode,
            influence_min=influence_min,
            influence_max=influence_max,
            expert_ids=expert_ids,
            smooth_replacement=smooth_replacement,
        )

        out = dict(
            pipeline=dict(
                input_json=str(json_path),
                step2_json=str(step2_path),
                predecessor_id=predecessor_id,
                new_id=new_id,
                alpha=alpha,
                expert_ids=expert_ids,
            ),
            consensus=dict(
                uniform=dict(level=consensus_uniform, compatibilities={expert_ids[i]: float(compat_uniform[i]) for i in range(len(expert_ids))}),
                initial=(
                    None
                    if consensus_initial is None
                    else dict(level=consensus_initial, weights={expert_ids[i]: float(initial_influence[i]) for i in range(len(expert_ids))})
                ),
                after=dict(level=consensus_after, compatibilities={expert_ids[i]: float(compat_after[i]) for i in range(len(expert_ids))}),
            ),
            dhf=dict(
                method=dhf_res.method,
                best_consensus=float(dhf_res.best_consensus),
                influence={expert_ids[i]: float(influence[i]) for i in range(len(expert_ids))},
                compatibility={dhf_res.dm_ids[i]: float(dhf_res.compatibility[i]) for i in range(len(dhf_res.dm_ids))}
                if dhf_res.compatibility
                else {},
            ),
            hives=dict(
                gamma=[float(x) for x in res_hives["gamma"]],
                gamma_scaled=[float(x) for x in res_hives["gamma_scaled"]],
                alt_scores=[float(x) for x in res_hives["alt_scores"]],
                ranking_1_based=[int(x) for x in (res_hives["ranking"] + 1)],
            ),
            lambdas=dict(
                base=_lambdas_to_dict(res_hives.get("lambdas_base"), expert_ids, problem.criteria),
                final=_lambdas_to_dict(res_hives.get("lambdas"), expert_ids, problem.criteria),
            ),
        )

        Path(out_path).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved HIVES+{method} result to: {out_path}")
        return out

    out_ga = _run_one("GA", out_ga_path)
    out_hho = _run_one("HHO", out_hho_path)

    return dict(step2_path=step2_path, ga=out_ga_path, hho=out_hho_path)


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


def _generate_dhf_payload(criteria_names: list[str], dm_ids: list[str], seed: int = 42) -> dict:
    """
    Generate DHF payload compatible with hives_dhf.dhf_consensus (criteria + dms[].pairwise_comparisons).
    Deterministic by seed.
    """
    rng = np.random.default_rng(int(seed))

    # Same scale as legacy/main4.py generator
    ifn = [
        [0.05, 0.95],
        [0.15, 0.80],
        [0.30, 0.60],
        [0.50, 0.50],
        [0.70, 0.20],
        [0.85, 0.10],
        [0.95, 0.05],
    ]
    pick = [0, 1, 2, 4, 5, 6]

    roles = [
        "Project Manager",
        "Sustainability Manager",
        "Investment Director",
        "Technical Expert",
        "Financial Analyst",
        "Quality Assurance",
    ]

    dms = []
    for dm_idx, dm_id in enumerate(dm_ids):
        comparisons: dict[str, dict[str, dict[str, list[float]]]] = {}
        for i, ci in enumerate(criteria_names):
            comparisons[ci] = {}
            for j, cj in enumerate(criteria_names):
                if i == j:
                    comparisons[ci][cj] = {"membership": [0.5], "non_membership": [0.5]}
                else:
                    idx = int(rng.choice(pick))
                    m_val = float(ifn[idx][0])
                    n_val = float(ifn[idx][1])
                    # Ensure m+n <= 1
                    if m_val + n_val > 1.0:
                        s = m_val + n_val
                        m_val /= s
                        n_val /= s
                    comparisons[ci][cj] = {"membership": [m_val], "non_membership": [n_val]}

        dms.append(
            dict(
                id=str(dm_id),
                role=roles[dm_idx % len(roles)],
                pairwise_comparisons=comparisons,
            )
        )

    return dict(
        problem_description=f"Generated DHFS data with {len(criteria_names)} criteria and {len(dm_ids)} experts",
        criteria=list(criteria_names),
        dms=dms,
        parameters=dict(
            desired_consensus=0.907,
            population_size=20,
            max_iterations=200,
            SearchAgents_no=10,
            Max_iter=100,
        ),
    )


def _generate_hives_payload(
    criteria_names: list[str],
    dm_ids: list[str],
    alternatives: list[str],
    seed: int = 42,
) -> dict:
    """
    Generate HIVES payload compatible with hives_dhf.json_input.load_decision_problem_from_data().
    """
    rng = np.random.default_rng(int(seed))
    n_alt = len(alternatives)
    n_crit = len(criteria_names)

    # Generate 1 DM score matrix per DM_id (alt x crit), then HIVES will aggregate by mean
    dms = []
    for dm_id in dm_ids:
        scores = rng.integers(low=1, high=101, size=(n_alt, n_crit)).tolist()
        dms.append(dict(id=str(dm_id), scores=scores))

    # Generate per-expert criteria weights (sum=100)
    experts = []
    for dm_id in dm_ids:
        w = rng.random(n_crit)
        w = (w / w.sum()) * 100.0
        experts.append(dict(id=str(dm_id), weights=[float(x) for x in w]))

    return dict(
        alternatives=list(alternatives),
        criteria=[dict(name=c, type="positive") for c in criteria_names],
        dms=dms,
        experts=experts,
        parameters=dict(alpha=0.95, B=2),
    )


def run_generated_compat3_hives_dhf_compat(
    n_criteria: int = 8,
    n_experts: int = 4,
    n_alternatives: int = 3,
    seed: int = 42,
    dhf_method: str = "HHO",
    output_path: str | None = None,
    generated_input_path: str | None = None,
) -> dict:
    """
    Порядок выполнения:
      1) Генератор данных (HIVES+ D(H)F)
      2) compat3 (равномерные веса)
      3) HIVES (равномерное влияние экспертов)
      4) DHF (оптимизация весов экспертов)
      5) compat3 (оптимизированные веса)
    Печатает compat3 до/после DHF.
    """
    n_criteria = int(n_criteria)
    n_experts = int(n_experts)
    n_alternatives = int(n_alternatives)
    seed = int(seed)

    if n_criteria <= 0 or n_experts <= 0 or n_alternatives <= 0:
        raise ValueError("n_criteria, n_experts, n_alternatives must be positive")

    dhf_method = str(dhf_method).upper()
    if dhf_method not in ("GA", "HHO"):
        raise ValueError("dhf_method must be GA or HHO")

    criteria_names = [f"Criterion_{i+1}" for i in range(n_criteria)]
    expert_ids = [f"DM{i+1}" for i in range(n_experts)]
    alternatives = [f"A{i+1}" for i in range(n_alternatives)]

    # 1) GENERATE INPUT
    hives_data = _generate_hives_payload(criteria_names, expert_ids, alternatives, seed=seed)
    dhf_data = _generate_dhf_payload(criteria_names, expert_ids, seed=seed)

    combined = dict(
        hives=hives_data,
        dhf=dhf_data,
        combined_parameters=dict(dhf_method=dhf_method),
    )

    if generated_input_path:
        Path(generated_input_path).write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved generated input to: {generated_input_path}")

    # Prepare matrices
    jm, jn, _crit, dm_ids = dhf_json_to_matrices(dhf_data)
    uniform_w = np.ones(len(dm_ids), dtype=float) / len(dm_ids)

    print("=" * 80)
    print("EXECUTION ORDER: GENERATOR -> compat3 -> HIVES -> DHF -> compat3")
    print("=" * 80)
    print(f"Generated: {n_criteria} criteria, {n_experts} experts, {n_alternatives} alternatives (seed={seed})")
    print(f"DHF method: {dhf_method}")

    # 2) compat3 (uniform)
    print("\n" + "=" * 80)
    print("[STEP 2] compat3 (uniform weights)")
    print("=" * 80)
    compat_before = compat3(uniform_w, jm, jn)
    cons_before = float(min(compat_before))
    print(f"Consensus level (min): {cons_before:.6f}")
    print(f"Compatibilities: {[f'{c:.6f}' for c in compat_before]}")

    # 3) HIVES (uniform influence)
    print("\n" + "=" * 80)
    print("[STEP 3] HIVES (uniform expert influence)")
    print("=" * 80)
    problem = load_decision_problem_from_data(hives_data)
    A = problem.aggregated_performance()
    W = problem.weights_matrix()
    uniform_influence = np.ones(len(expert_ids), dtype=float) / len(expert_ids)
    res_hives = hives_rank(A=A, W=W, influence=uniform_influence, expert_ids=expert_ids)

    print("\n[HIVES] gamma_scaled (sum=100):")
    print(np.round(res_hives["gamma_scaled"], 2))
    print("[HIVES] ranking (1-based):")
    print(res_hives["ranking"] + 1)

    # 4) DHF (optimize weights)
    print("\n" + "=" * 80)
    print("[STEP 4] DHF (optimization of expert weights)")
    print("=" * 80)
    dhf_res = optimize_expert_weights(dhf_data, method=dhf_method)
    print(f"Optimized expert weights: {[f'{w:.4f}' for w in dhf_res.best_weights]}")
    print(f"Best consensus (from optimizer): {dhf_res.best_consensus:.6f}")

    # 5) compat3 (optimized)
    print("\n" + "=" * 80)
    print("[STEP 5] compat3 (optimized weights)")
    print("=" * 80)
    compat_after = compat3(dhf_res.best_weights, jm, jn)
    cons_after = float(min(compat_after))
    print(f"Consensus level (min): {cons_after:.6f}")
    print(f"Compatibilities: {[f'{c:.6f}' for c in compat_after]}")

    out = dict(
        generated=dict(
            n_criteria=n_criteria,
            n_experts=n_experts,
            n_alternatives=n_alternatives,
            seed=seed,
            criteria=criteria_names,
            expert_ids=expert_ids,
            dhf_method=dhf_method,
        ),
        compat3_uniform=dict(consensus=cons_before, compatibilities=[float(x) for x in compat_before]),
        hives=dict(
            ranking_1_based=[int(x) for x in (res_hives["ranking"] + 1)],
            gamma_scaled=[float(x) for x in res_hives["gamma_scaled"]],
        ),
        dhf=dict(
            dm_ids=dhf_res.dm_ids,
            weights=[float(x) for x in dhf_res.best_weights],
            best_consensus=float(dhf_res.best_consensus),
        ),
        compat3_optimized=dict(consensus=cons_after, compatibilities=[float(x) for x in compat_after]),
    )

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

    p_gen_pipeline = sub.add_parser(
        "gen-compat-hives-dhf-compat",
        help="Генератор -> compat3 -> HIVES -> DHF -> compat3 (печать compat3 до/после)",
    )
    p_gen_pipeline.add_argument("--criteria", type=int, default=8, help="Количество критериев (по умолчанию 8).")
    p_gen_pipeline.add_argument("--experts", type=int, default=4, help="Количество экспертов/DM (по умолчанию 4).")
    p_gen_pipeline.add_argument(
        "--alternatives", type=int, default=3, help="Количество альтернатив (по умолчанию 3)."
    )
    p_gen_pipeline.add_argument("--seed", type=int, default=42, help="Seed для генерации (по умолчанию 42).")
    p_gen_pipeline.add_argument(
        "--dhf-method", type=str, default="HHO", help="Метод оптимизации весов: GA или HHO (по умолчанию HHO)."
    )
    p_gen_pipeline.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Опционально: сохранить итоговый результат (JSON).",
    )
    p_gen_pipeline.add_argument(
        "--save-input",
        type=str,
        default=None,
        help="Опционально: сохранить сгенерированный combined JSON (HIVES+ D(H)F).",
    )

    p_pipe = sub.add_parser(
        "pipeline",
        help="Pipeline: (step2=remove expert)->compat3->GA+HHO->HIVES, save separate files.",
    )
    p_pipe.add_argument("json_path", type=str, help="Путь к combined JSON (ожидается rotation+combined_parameters).")

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
    elif args.cmd == "gen-compat-hives-dhf-compat":
        run_generated_compat3_hives_dhf_compat(
            n_criteria=args.criteria,
            n_experts=args.experts,
            n_alternatives=args.alternatives,
            seed=args.seed,
            dhf_method=args.dhf_method,
            output_path=args.output,
            generated_input_path=args.save_input,
        )
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
    elif args.cmd == "pipeline":
        run_pipeline_replace_ga_hho(args.json_path)
    else:
        raise SystemExit(
            "Usage: python main.py hives input.json  OR  python main.py combined combined.json  OR  python main.py experiment combined.json  OR  python main.py compare-order combined.json  OR  python main.py hives-compat-dhf combined.json  OR  python main.py gen-compat-hives-dhf-compat"
        )

