from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np


CriterionType = Literal["positive", "negative"]


@dataclass
class Criterion:
    name: str
    type: CriterionType = "positive"


@dataclass
class DecisionMaker:
    id: str
    # матрица (альтернатива × критерий)
    scores: np.ndarray


@dataclass
class Expert:
    id: str
    # веса критериев для эксперта (одна строка W)
    weights: np.ndarray


@dataclass
class GroupParameters:
    alpha: float = 0.95
    B: int = 2


@dataclass
class DecisionProblem:
    alternatives: List[str]
    criteria: List[Criterion]
    dms: List[DecisionMaker]
    parameters: GroupParameters
    # эксперты, задающие веса критериев (матрица W)
    experts: Optional[List[Expert]] = None

    def aggregated_performance(self) -> np.ndarray:
        # Агрегация оценок ЛПР: усреднённая матрица A (альтернатива × критерий).
        if not self.dms:
            raise ValueError("DecisionProblem must contain at least one decision maker")

        dm_matrices = [dm.scores for dm in self.dms]
        stacked = np.stack(dm_matrices, axis=0)  # (dm, alt, crit)
        return stacked.mean(axis=0)

    def weights_matrix(self) -> np.ndarray:
        # Матрица W (эксперт × критерий): из JSON или равные веса по умолчанию.
        num_criteria = len(self.criteria)
        if self.experts:
            rows = [np.asarray(e.weights, dtype=float) for e in self.experts]
            W = np.vstack(rows)
            if W.shape[1] != num_criteria:
                raise ValueError("Each expert.weights must have length = number of criteria")
            return W

        # fallback: равные веса по критериям, по одному «эксперту» на ЛПР
        return build_equal_weights(num_experts=len(self.dms), num_criteria=num_criteria)


def build_equal_weights(num_experts: int, num_criteria: int) -> np.ndarray:
    # Равномерные веса критериев (сумма каждой строки W равна 100).
    if num_criteria <= 0:
        raise ValueError("num_criteria must be positive")

    row = np.full(num_criteria, 100.0 / num_criteria, dtype=float)
    W = np.tile(row, (num_experts, 1))
    return W


