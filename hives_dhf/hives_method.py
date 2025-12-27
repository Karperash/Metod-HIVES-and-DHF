import math
from typing import Tuple, List, Dict, Any, Optional

import numpy as np


def clamp_and_normalize_influence(
    influence: np.ndarray,
    min_w: float = 0.01,
    max_w: float = 0.99,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Social constraints from тезисы (упрощённый вариант):
    - clamp each influence to [min_w, max_w]
    - normalize so sum = 1
    """
    w = np.asarray(influence, dtype=float).copy()
    w = np.clip(w, min_w, max_w)
    s = float(w.sum())
    if s <= eps:
        w[:] = 1.0 / len(w)
        return w
    return w / s


def smooth_replace_lambdas(
    lambdas_base: np.ndarray,
    expert_ids: List[str],
    predecessor_id: str,
    new_id: str,
    alpha: float,
    predecessor_old_lambdas_row: np.ndarray,
    col_sum: float = 100.0,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    "Плавная замена" эксперта из тезисов, применяемая к матрице λ (в % по каждому критерию).

    lambdas_base: shape (m_experts, n_criteria), col-sum ~= 100
    predecessor_old_lambdas_row: shape (n_criteria,), веса ушедшего эксперта в прошлой группе (в %)

    Формула:
      λ_final[n,c] = α*λ_base[n,c] + (1-α)*λ_old[p,c]
      λ_final[s,c] = λ_base[s,c] * (col_sum - λ_final[n,c]) / (col_sum - λ_base[n,c])  for s != n
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError("alpha must be in [0, 1]")

    lambdas_base = np.asarray(lambdas_base, dtype=float)
    pred_old = np.asarray(predecessor_old_lambdas_row, dtype=float).reshape(-1)
    if lambdas_base.ndim != 2:
        raise ValueError("lambdas_base must be 2D")
    m, n = lambdas_base.shape
    if pred_old.shape[0] != n:
        raise ValueError("predecessor_old_lambdas_row must have length = number of criteria")

    try:
        new_idx = expert_ids.index(new_id)
    except ValueError as e:
        raise ValueError(f"new_id {new_id!r} not found in expert_ids") from e

    # predecessor_id is expected to be absent from current group (replaced),
    # but we still accept it for logging / clarity.
    _ = predecessor_id

    out = lambdas_base.copy()
    out[new_idx, :] = alpha * lambdas_base[new_idx, :] + (1.0 - alpha) * pred_old

    # rescale others per criterion to keep column sums
    for c in range(n):
        denom = (col_sum - lambdas_base[new_idx, c])
        numer = (col_sum - out[new_idx, c])
        if abs(denom) <= eps:
            # fallback: keep others, then renormalize the column
            pass
        else:
            factor = numer / denom
            for s in range(m):
                if s == new_idx:
                    continue
                out[s, c] = lambdas_base[s, c] * factor

        # final renormalize this column to exactly col_sum (numerical stability)
        col = out[:, c]
        col_s = float(col.sum())
        if col_s > eps:
            out[:, c] = col * (col_sum / col_s)
        else:
            out[:, c] = col_sum / m

    return out


def score_bell_column(
    weights: np.ndarray,
    s_mean: float = 50.0,
    s_max: float = 100.0,
    shape: float = 2.0,
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float]]:
    # Score Bell для одного критерия; возвращает SB-баллы и параметры колокола.
    w = np.asarray(weights, dtype=float)
    xmin = float(w.min())
    xmax = float(w.max())
    Q1 = float(np.percentile(w, 25))
    Q3 = float(np.percentile(w, 75))

    # SICP = среднее по значениям внутри [Q1, Q3]
    mask_iz = (w >= Q1) & (w <= Q3)
    sicp = float(w[mask_iz].mean())

    scores = np.empty_like(w, dtype=float)

    for i, val in enumerate(w):
        # Зона 1: [min, Q1)
        if val < Q1:
            xrel = val - xmin
            denom = Q1 - xmin or 1e-9
            scores[i] = math.exp(math.log(s_mean) * (xrel / denom) ** shape)

        # Зона 2: [Q1, SICP)
        elif val < sicp:
            xrel = val - Q1
            denom = sicp - Q1 or 1e-9
            scores[i] = (s_max + 1) - math.exp(
                math.log(s_max + 1 - s_mean) * (1 - xrel / denom) ** shape
            )

        # Зона 3: [SICP, Q3]
        elif val <= Q3:
            xrel = val - sicp
            denom = Q3 - sicp or 1e-9
            scores[i] = (s_max + 1) - math.exp(
                math.log(s_max + 1 - s_mean) * (xrel / denom) ** shape
            )

        # Зона 4: (Q3, max]
        else:
            xrel = val - Q3
            denom = xmax - Q3 or 1e-9
            scores[i] = math.exp(math.log(s_mean) * (1 - xrel / denom) ** shape)

    return scores, (xmin, Q1, sicp, Q3, xmax)


def expand_by_influence(
    W: np.ndarray,
    influence: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Vote-casting: раздуваем матрицу W, повторяя строки пропорционально влиянию.
    W = np.asarray(W, float)
    influence = np.asarray(influence, float)

    if influence.shape[0] != W.shape[0]:
        raise ValueError("influence length must match number of rows in W")
    if np.any(influence <= 0):
        raise ValueError("influence values must be positive")

    alpha_min = float(influence.min())
    # число голосов = влияние / минимальное влияние
    votes = np.rint(influence / alpha_min).astype(int)

    owners = np.concatenate([np.full(v, k, dtype=int) for k, v in enumerate(votes)])
    W_exp = np.repeat(W, votes, axis=0)
    return W_exp, owners, votes


def hives(
    W: np.ndarray,
    influence: np.ndarray | None = None,
    influence_mode: str = "continuous",
    influence_min: float = 0.01,
    influence_max: float = 0.99,
    s_mean: float = 50.0,
    s_max: float = 100.0,
    shape: float = 2.0,
    expert_ids: Optional[List[str]] = None,
    smooth_replacement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Основной расчёт HIVES для матрицы W (влияния экспертов и Score Bell).
    W = np.asarray(W, float)
    m, n = W.shape

    # Если влияние не задано — все эксперты равны
    if influence is None:
        W_exp = W
        owners = np.arange(m, dtype=int)
    else:
        influence = clamp_and_normalize_influence(influence, min_w=influence_min, max_w=influence_max)
        if influence_mode == "discrete":
            W_exp, owners, _votes = expand_by_influence(W, influence)
        elif influence_mode == "continuous":
            W_exp = W
            owners = np.arange(m, dtype=int)
        else:
            raise ValueError("influence_mode must be 'continuous' or 'discrete'")

    # Score Bell на "раздутой" матрице
    S_exp = np.zeros_like(W_exp)
    params = []
    for j in range(n):
        scores_j, params_j = score_bell_column(
            W_exp[:, j], s_mean=s_mean, s_max=s_max, shape=shape
        )
        S_exp[:, j] = scores_j
        params.append(params_j)

    # Складываем баллы обратно по исходным экспертам
    S = np.zeros((m, n))
    for i in range(m):
        S[i, :] = S_exp[owners == i, :].sum(axis=0)

    # Непрерывное влияние: масштабируем вклады экспертов перед нормировкой λ
    if influence is not None and influence_mode == "continuous":
        S = S * influence.reshape(-1, 1)

    # Веса экспертов по критериям (λ_ij)
    col_sums = S.sum(axis=0, keepdims=True)
    lambdas = S / col_sums * 100

    lambdas_base = lambdas.copy()

    # Плавная замена эксперта (опционально)
    if smooth_replacement is not None:
        if expert_ids is None:
            raise ValueError("expert_ids must be provided when using smooth_replacement")
        predecessor_id = str(smooth_replacement["predecessor_id"])
        new_id = str(smooth_replacement["new_id"])
        alpha = float(smooth_replacement["alpha"])
        pred_old_row = np.asarray(smooth_replacement["predecessor_old_lambdas_row"], dtype=float)
        lambdas = smooth_replace_lambdas(
            lambdas_base=lambdas_base,
            expert_ids=expert_ids,
            predecessor_id=predecessor_id,
            new_id=new_id,
            alpha=alpha,
            predecessor_old_lambdas_row=pred_old_row,
            col_sum=100.0,
        )

    # Веса критериев γ_j
    gamma = (lambdas * W).sum(axis=0) / 100

    # Масштабируем до суммы 100
    beta = 100.0 / float(gamma.sum())
    gamma_scaled = gamma * beta

    return dict(
        gamma=gamma,
        gamma_scaled=gamma_scaled,
        lambdas=lambdas,
        lambdas_base=lambdas_base,
        scores=S,
        params=params,
        influence=(None if influence is None else np.asarray(influence, float)),
    )


def hives_rank(
    A: np.ndarray,
    W: np.ndarray,
    influence: np.ndarray | None = None,
    influence_mode: str = "continuous",
    influence_min: float = 0.01,
    influence_max: float = 0.99,
    s_mean: float = 50.0,
    s_max: float = 100.0,
    shape: float = 2.0,
    expert_ids: Optional[List[str]] = None,
    smooth_replacement: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # Обёртка над hives(): считает итоговые оценки и ранжирование альтернатив.
    A = np.asarray(A, float)
    res = hives(
        W,
        influence=influence,
        influence_mode=influence_mode,
        influence_min=influence_min,
        influence_max=influence_max,
        s_mean=s_mean,
        s_max=s_max,
        shape=shape,
        expert_ids=expert_ids,
        smooth_replacement=smooth_replacement,
    )
    gamma_scaled = res["gamma_scaled"]

    # Вклад по каждому критерию
    alt_details = A * gamma_scaled  # по статье просто умножают
    alt_scores = alt_details.sum(axis=1)
    ranking = np.argsort(-alt_scores)  # по убыванию

    res.update(
        dict(
            alt_scores=alt_scores,
            alt_details=alt_details,
            ranking=ranking,
        )
    )
    return res


