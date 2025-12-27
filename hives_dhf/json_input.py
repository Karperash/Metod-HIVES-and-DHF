from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from .models import Criterion, DecisionMaker, GroupParameters, DecisionProblem, Expert


def load_decision_problem_from_data(data: Dict[str, Any]) -> DecisionProblem:
    # Чтение задачи из dict (для combined JSON): альтернативы, критерии, матрица A и матрица W.
    alternatives = list(data["alternatives"])

    criteria = [
        Criterion(name=c["name"], type=c.get("type", "positive"))
        for c in data["criteria"]
    ]

    dms: List[DecisionMaker] = []
    for dm_obj in data["dms"]:
        scores = np.array(dm_obj["scores"], dtype=float)
        dms.append(DecisionMaker(id=dm_obj["id"], scores=scores))

    experts: Optional[List[Expert]] = None
    if "experts" in data:
        experts = []
        for e_obj in data["experts"]:
            weights = np.array(e_obj["weights"], dtype=float)
            experts.append(Expert(id=e_obj["id"], weights=weights))

    params_data = data.get("parameters", {})
    parameters = GroupParameters(
        alpha=float(params_data.get("alpha", 0.95)),
        B=int(params_data.get("B", 2)),
    )

    return DecisionProblem(
        alternatives=alternatives,
        criteria=criteria,
        dms=dms,
        parameters=parameters,
        experts=experts,
    )


def load_decision_problem_from_json(path: str | Path) -> DecisionProblem:
    # Чтение задачи из JSON: альтернативы, критерии, матрица A и матрица W.
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    return load_decision_problem_from_data(data)


