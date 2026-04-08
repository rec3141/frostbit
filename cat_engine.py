"""
Computerized Adaptive Testing (CAT) engine.

Uses a simplified IRT model where each item has a single difficulty parameter
(derived from ensemble model calibration + Haiku/Sonnet ratings).

The engine tracks two ability estimates:
- a full Bayesian posterior theta used for scoring and reporting
- a recency-weighted routing theta used only to choose the next item
"""

import math
import random
from dataclasses import dataclass, field


@dataclass
class Item:
    id: int
    question: str
    choices: dict  # {"A": "...", "B": "...", ...}
    answer: str
    difficulty: float  # calibrated difficulty on a continuous scale
    tier: str  # "easy", "medium", "hard" — original generation label
    page_ref: str = ""
    blooms_level: int = 0
    blooms_name: str = ""


@dataclass
class Response:
    item_id: int
    chosen: str
    correct: bool
    theta_after: float
    se_after: float
    skipped: bool = False


@dataclass
class CATSession:
    student_id: str
    items_pool: list  # available Item objects
    responses: list = field(default_factory=list)
    theta: float = -2.0  # start low — student works up
    se: float = 2.0  # high initial uncertainty
    _used_ids: set = field(default_factory=set)
    current_item_id: int | None = None
    finished: bool = False
    resume_token: str = ""
    initial_theta: float = -2.0
    recency_half_life: float = 8.0
    routing_theta: float = -2.0
    routing_se: float = 2.0
    skip_budget: int = 0
    skip_count: int = 0

    # Config
    max_questions: int = 30
    min_questions: int = 10
    se_threshold: float = 0.3  # stop when SE drops below this


def icc(theta: float, difficulty: float) -> float:
    """Item Characteristic Curve — 1PL/Rasch model.
    Returns P(correct | theta, difficulty).
    """
    logit = theta - difficulty
    # Clamp to avoid overflow
    logit = max(-10, min(10, logit))
    return 1.0 / (1.0 + math.exp(-logit))


def item_information(theta: float, difficulty: float) -> float:
    """Fisher information for a 1PL item at a given theta.
    I(theta) = P(theta) * (1 - P(theta))
    """
    p = icc(theta, difficulty)
    return p * (1 - p)


THETA_GRID_MIN = -4.0
THETA_GRID_MAX = 4.0
THETA_GRID_STEP = 0.05
THETA_GRID = [
    THETA_GRID_MIN + (i * THETA_GRID_STEP)
    for i in range(int((THETA_GRID_MAX - THETA_GRID_MIN) / THETA_GRID_STEP) + 1)
]


def estimate_posterior_theta(
    responses: list[tuple[float, bool]],
    prior_mean: float = -2.0,
    prior_var: float = 2.0,
) -> tuple[float, float]:
    """Estimate theta from the full Bayesian posterior over a fixed grid."""
    if not responses:
        return prior_mean, math.sqrt(prior_var)

    log_probs = []
    for theta in THETA_GRID:
        log_prob = -0.5 * (((theta - prior_mean) ** 2) / prior_var)
        for difficulty, correct in responses:
            p = max(1e-9, min(1 - 1e-9, icc(theta, difficulty)))
            log_prob += math.log(p if correct else (1 - p))
        log_probs.append(log_prob)

    max_log_prob = max(log_probs)
    probs = [math.exp(lp - max_log_prob) for lp in log_probs]
    total_prob = sum(probs)
    if total_prob <= 0:
        return prior_mean, math.sqrt(prior_var)

    posterior_mean = sum(theta * prob for theta, prob in zip(THETA_GRID, probs)) / total_prob
    posterior_var = sum(
        ((theta - posterior_mean) ** 2) * prob
        for theta, prob in zip(THETA_GRID, probs)
    ) / total_prob
    return posterior_mean, math.sqrt(max(posterior_var, 1e-10))


def estimate_routing_theta(
    responses: list[tuple[float, bool]],
    prior_mean: float = -2.0,
    prior_var: float = 2.0,
    recency_half_life: float = 8.0,
) -> tuple[float, float]:
    """Estimate a fast-moving routing theta via recency-weighted MAP."""
    if not responses:
        return prior_mean, math.sqrt(prior_var)

    # Recency weighting: exponential decay so recent answers dominate
    # Half-life of ~8 questions: answer from 8 Qs ago has half the weight
    n = len(responses)
    half_life = max(1.0, float(recency_half_life))
    decay = 0.5 ** (1.0 / half_life)
    weights = [decay ** (n - 1 - i) for i in range(n)]

    # Start Newton-Raphson from a reasonable point
    # Use weighted proportion correct as initial guess
    w_correct = sum(w for w, (_, c) in zip(weights, responses) if c)
    w_total = sum(weights)
    prop = w_correct / w_total
    prop = max(0.05, min(0.95, prop))
    theta = math.log(prop / (1 - prop))

    for _ in range(50):
        grad = 0.0
        hessian = 0.0
        for w, (diff, correct) in zip(weights, responses):
            p = icc(theta, diff)
            grad += w * ((1.0 if correct else 0.0) - p)
            hessian -= w * p * (1 - p)

        # Prior terms
        grad -= (theta - prior_mean) / prior_var
        hessian -= 1.0 / prior_var

        if abs(hessian) < 1e-10:
            break

        step = grad / hessian
        step = max(-0.5, min(0.5, step))
        theta -= step

        if abs(step) < 1e-6:
            break

    theta = max(-3.0, min(3.0, theta))

    # SE from inverse Fisher information
    total_info = sum(w * item_information(theta, d)
                     for w, (d, _) in zip(weights, responses))
    total_info += 1.0 / prior_var
    se = 1.0 / math.sqrt(max(total_info, 1e-10))

    return theta, se


def estimate_theta(
    responses: list[tuple[float, bool]],
    prior_mean: float = -2.0,
    prior_var: float = 2.0,
    recency_half_life: float = 8.0,
) -> tuple[float, float]:
    """Backward-compatible alias for the reporting posterior theta."""
    return estimate_posterior_theta(responses, prior_mean=prior_mean, prior_var=prior_var)


def select_next_item(session: CATSession) -> Item | None:
    """Select the next item matching current ability estimate.

    Uses maximum information criterion: pick items whose difficulty
    is closest to current theta (where the model learns most).
    Jitters among top candidates for variety.
    """
    available = [item for item in session.items_pool
                 if item.id not in session._used_ids]

    if not available:
        return None

    # Target difficulty follows the short-memory routing estimate.
    target = session.routing_theta

    # Sort by closeness to target difficulty
    available.sort(key=lambda item: abs(item.difficulty - target))

    # Pick randomly from top 5 closest to add variety
    top_k = min(5, len(available))
    return random.choice(available[:top_k])


def process_answer(session: CATSession, item: Item, chosen: str) -> Response:
    """Process a student's answer and update the session."""
    if session.finished:
        raise ValueError("session is already finished")
    if session.current_item_id != item.id:
        raise ValueError("unexpected item")
    if item.id in session._used_ids:
        raise ValueError("item already answered")

    correct = chosen.upper() == item.answer.upper()
    session._used_ids.add(item.id)
    resp_history = []
    for r in session.responses:
        if r.skipped:
            continue
        # Find item difficulty from pool
        for it in session.items_pool:
            if it.id == r.item_id:
                resp_history.append((it.difficulty, r.correct))
                break
    resp_history.append((item.difficulty, correct))

    # Posterior theta is used for scoring and reporting.
    session.theta, session.se = estimate_posterior_theta(
        resp_history,
        prior_mean=session.initial_theta,
    )

    # Routing theta is used only for next-item selection.
    session.routing_theta, session.routing_se = estimate_routing_theta(
        resp_history,
        prior_mean=session.initial_theta,
        recency_half_life=session.recency_half_life,
    )

    response = Response(
        item_id=item.id,
        chosen=chosen,
        correct=correct,
        theta_after=session.theta,
        se_after=session.se,
    )
    session.responses.append(response)
    session.current_item_id = None
    return response


def skip_item(session: CATSession, item: Item) -> Response:
    """Skip the current item without affecting scoring."""
    if session.finished:
        raise ValueError("session is already finished")
    if session.current_item_id != item.id:
        raise ValueError("unexpected item")
    if item.id in session._used_ids:
        raise ValueError("item already answered")
    if session.skip_count >= session.skip_budget:
        raise ValueError("skip budget exhausted")

    session._used_ids.add(item.id)
    session.skip_count += 1
    response = Response(
        item_id=item.id,
        chosen="SKIP",
        correct=False,
        theta_after=session.theta,
        se_after=session.se,
        skipped=True,
    )
    session.responses.append(response)
    session.current_item_id = None
    return response


def should_stop(session: CATSession) -> bool:
    """Determine if the test should stop. Only stops if student requests it."""
    return False


def theta_to_percent(theta: float) -> float:
    """Convert theta (logit) to an approximate percentage score.
    Maps the typical theta range [-3, 3] to [0, 100].
    """
    # Sigmoid mapping
    p = 1.0 / (1.0 + math.exp(-theta))
    return round(p * 100, 1)


def get_performance_summary(session: CATSession) -> dict:
    """Generate a summary of the student's performance."""
    scored_responses = [r for r in session.responses if not r.skipped]
    n = len(scored_responses)
    correct = sum(1 for r in scored_responses if r.correct)
    skipped = sum(1 for r in session.responses if r.skipped)

    # Difficulty breakdown
    tier_stats = {}
    for r in scored_responses:
        item = next((it for it in session.items_pool if it.id == r.item_id), None)
        if item:
            tier = item.tier
            if tier not in tier_stats:
                tier_stats[tier] = {"total": 0, "correct": 0}
            tier_stats[tier]["total"] += 1
            if r.correct:
                tier_stats[tier]["correct"] += 1

    return {
        "total_questions": n,
        "correct": correct,
        "raw_percent": round(100 * correct / n, 1) if n else 0,
        "theta": round(session.theta, 2),
        "se": round(session.se, 2),
        "estimated_ability": theta_to_percent(session.theta),
        "skipped": skipped,
        "skip_budget": session.skip_budget,
        "skip_remaining": max(0, session.skip_budget - session.skip_count),
        "tier_breakdown": tier_stats,
        "theta_history": [
            {"question": i + 1, "theta": round(r.theta_after, 2), "correct": r.correct}
            for i, r in enumerate(scored_responses)
        ],
    }
