"""
Answer Readiness Score (ARS) -- StreamReady paper Sec. 4.3, Eqs. 9-11.

ARS scores *when* a model answered relative to the window in which the supporting
evidence is actually visible. Each question carries a ground-truth evidence window
``[t_s, t_e]``: ``t_s`` is when sufficient evidence first appears, ``t_e`` when it
stops being valid. Given an answer time ``t_a``::

    ARS  = mean_i ( EP_i * LP_i )
    Acc_e = Acc * ARS                                                      (Eq. 9)

    EP = softmin( 1, 2 * sigmoid( gamma_e * (t_a - t_s) / (tau + eps) ) )  (Eq. 10)
    LP = softmin( 1, softmax( 0, 1 - gamma_l * (t_a - t_e) / (tau + eps) ))(Eq. 11)

The asymmetry is the point. Answering *before* any evidence exists is speculation,
so EP falls off sharply (gamma_e = 6). Answering *after* the window is mere
hesitation, so LP decays gently (gamma_l = 1). ``tau`` is the median evidence
duration over the evaluated set, which puts videos with very short and very long
windows on a comparable scale.

Details that the paper calls out explicitly and that are easy to get wrong:

* The **factor of 2** inside EP exists so that answering exactly at onset
  (``t_a == t_s``) scores 1.0 rather than the 0.5 a bare sigmoid would give.
* ``softmin``/``softmax`` here act as smooth clamps at the window boundaries,
  preventing abrupt jumps in score just inside or outside the window.
* Inside the window (``t_s < t_a < t_e``) both penalties are 1.
* LP is linear in the delay and reaches exactly 0 one median evidence duration
  past ``t_e`` (``1 - gamma_l * tau / tau`` with ``gamma_l = 1``). Within that
  range it is markedly milder than EP at the same offset -- at a quarter of a
  median duration, LP is 0.75 against EP's 0.36 -- which is the asymmetry the
  formulation is built around.
* If noisy annotation yields ``t_s > t_e``, monotonicity keeps ARS in [0, 1]
  rather than producing a contradictory or negative value.
* An unanswerable question is encoded as ``tau = t_s = t_e = 0``, which drives the
  product toward zero so effective accuracy is near zero even if an answer was
  produced.
* For multi-turn questions ARS is computed per turn and averaged.
"""

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

# Paper Sec. 5, "Implementation Details".
GAMMA_EARLY_DEFAULT = 6.0
GAMMA_LATE_DEFAULT = 1.0
EPS = 1e-6


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable for large |x|; EP saturates for very early answers.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def early_penalty(
    t_answer: np.ndarray,
    t_start: np.ndarray,
    tau: float,
    gamma_e: float = GAMMA_EARLY_DEFAULT,
) -> np.ndarray:
    """Eq. 10. 1.0 at or after evidence onset, decaying sharply before it."""
    z = gamma_e * (t_answer - t_start) / (tau + EPS)
    return np.minimum(1.0, 2.0 * _sigmoid(z))


def late_penalty(
    t_answer: np.ndarray,
    t_end: np.ndarray,
    tau: float,
    gamma_l: float = GAMMA_LATE_DEFAULT,
) -> np.ndarray:
    """Eq. 11. 1.0 up to the end of the window, decaying gently after it."""
    z = 1.0 - gamma_l * (t_answer - t_end) / (tau + EPS)
    return np.minimum(1.0, np.maximum(0.0, z))


def median_evidence_duration(records: Sequence[Dict]) -> float:
    """``tau``: median of ``t_e - t_s`` over the evaluated set.

    Unanswerable entries (``t_s == t_e == 0``) are excluded so they cannot drag
    the scale to zero for everyone else.
    """
    durs = [
        float(r["evidence_end"]) - float(r["evidence_start"])
        for r in records
        if not (float(r["evidence_start"]) == 0.0 and float(r["evidence_end"]) == 0.0)
    ]
    durs = [d for d in durs if d > 0]
    return float(np.median(durs)) if durs else 1.0


def compute_ars(
    records: Sequence[Dict],
    gamma_e: float = GAMMA_EARLY_DEFAULT,
    gamma_l: float = GAMMA_LATE_DEFAULT,
    tau: Optional[float] = None,
    breakdown: bool = True,
) -> Dict[str, float]:
    """Score a set of answered questions.

    Args:
        records: dicts with at least ``answer_time``, ``evidence_start``,
            ``evidence_end``. Optional keys: ``correct`` (bool/0-1) to get
            accuracy and effective accuracy, ``task`` for a per-task breakdown,
            and ``turn_group`` to average multi-turn questions per dialogue
            before aggregating.
        tau: median evidence duration; computed from ``records`` when omitted.
        breakdown: emit the per-task split. Set ``False`` on the recursive call,
            which otherwise re-enters forever because the sub-records still carry
            their ``task`` key.

    Returns:
        ``{"ars", "accuracy", "effective_accuracy", "n", "tau", "early_penalty",
        "late_penalty", "on_time_rate"}`` plus ``per_task`` when tasks are given.
    """
    if not records:
        return {"ars": 0.0, "accuracy": 0.0, "effective_accuracy": 0.0, "n": 0,
                "tau": 0.0, "early_penalty": 0.0, "late_penalty": 0.0,
                "on_time_rate": 0.0}

    if tau is None:
        tau = median_evidence_duration(records)

    ta = np.array([float(r["answer_time"]) for r in records], dtype=np.float64)
    ts = np.array([float(r["evidence_start"]) for r in records], dtype=np.float64)
    te = np.array([float(r["evidence_end"]) for r in records], dtype=np.float64)

    ep = early_penalty(ta, ts, tau, gamma_e)
    lp = late_penalty(ta, te, tau, gamma_l)
    per_q = ep * lp

    # Unanswerable: tau = t_s = t_e = 0 by convention, score 0.
    unanswerable = (ts == 0.0) & (te == 0.0)
    per_q = np.where(unanswerable, 0.0, per_q)

    # Multi-turn: average within a dialogue first so a long dialogue does not
    # outweigh a single-turn question (paper Sec. 4.3).
    groups = [r.get("turn_group") for r in records]
    if any(g is not None for g in groups):
        buckets: Dict[object, List[float]] = {}
        for g, v, i in zip(groups, per_q, range(len(records))):
            buckets.setdefault(g if g is not None else ("__single__", i), []).append(float(v))
        ars = float(np.mean([float(np.mean(v)) for v in buckets.values()]))
    else:
        ars = float(per_q.mean())

    out = {
        "ars": ars,
        "n": int(len(records)),
        "tau": float(tau),
        "early_penalty": float(ep.mean()),
        "late_penalty": float(lp.mean()),
        # Fraction answered strictly inside the window, a readable companion to ARS.
        "on_time_rate": float(np.mean((ta >= ts) & (ta <= te) & ~unanswerable)),
    }

    if all("correct" in r for r in records):
        acc = float(np.mean([float(bool(r["correct"])) for r in records]))
        out["accuracy"] = acc
        out["effective_accuracy"] = acc * ars            # Eq. 9

    tasks = {r.get("task") for r in records if r.get("task")}
    if breakdown and tasks:
        out["per_task"] = {}
        for t in sorted(tasks):
            sub = [r for r in records if r.get("task") == t]
            # Same tau across tasks so per-task numbers stay comparable, and
            # breakdown=False so this does not recurse.
            s = compute_ars(sub, gamma_e, gamma_l, tau, breakdown=False)
            out["per_task"][t] = {
                k: s[k] for k in ("ars", "accuracy", "effective_accuracy", "n")
                if k in s
            }
    return out


def format_ars_report(summary: Dict[str, float]) -> str:
    """Render :func:`compute_ars` output as a readable block."""
    lines = [
        f"questions          {summary.get('n', 0)}",
        f"tau (median dur.)  {summary.get('tau', 0.0):.2f}",
        f"ARS                {summary.get('ars', 0.0):.4f}",
        f"  early penalty    {summary.get('early_penalty', 0.0):.4f}",
        f"  late penalty     {summary.get('late_penalty', 0.0):.4f}",
        f"  on-time rate     {summary.get('on_time_rate', 0.0):.4f}",
    ]
    if "accuracy" in summary:
        lines += [
            f"Accuracy           {summary['accuracy']:.4f}",
            f"Effective accuracy {summary['effective_accuracy']:.4f}",
        ]
    if summary.get("per_task"):
        lines.append("per task:")
        for t, s in summary["per_task"].items():
            acc = f" acc {s['accuracy']:.4f}" if "accuracy" in s else ""
            eff = f" acc_e {s['effective_accuracy']:.4f}" if "effective_accuracy" in s else ""
            lines.append(f"  {t:<5} n {s['n']:<5} ARS {s['ars']:.4f}{acc}{eff}")
    return "\n".join(lines)
