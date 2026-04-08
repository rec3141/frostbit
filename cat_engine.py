"""
Computerized Adaptive Testing (CAT) engine.

Uses a simplified IRT model where each item has a single difficulty parameter
(derived from ensemble model calibration + Haiku/Sonnet ratings).

Ability (theta) is estimated via maximum likelihood after each response.
Items are selected to maximize information at the current theta estimate.
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


@dataclass
class CATSession:
    student_id: str
    items_pool: list  # available Item objects
    responses: list = field(default_factory=list)
    theta: float = -2.0  # start low — student works up
    se: float = 2.0  # high initial uncertainty
    _used_ids: set = field(default_factory=set)

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


def estimate_theta(responses: list[tuple[float, bool]],
                   prior_mean: float = -2.0,
                   prior_var: float = 2.0) -> tuple[float, float]:
    """Estimate ability (theta) via maximum a posteriori (MAP).

    Uses a recency-weighted approach: recent responses count more than
    older ones, so the estimate adapts to the student's current level
    rather than being anchored by early easy questions.

    Args:
        responses: list of (difficulty, correct) tuples
        prior_mean: prior mean for theta (normal prior)
        prior_var: prior variance for theta

    Returns:
        (theta_hat, standard_error)
    """
    if not responses:
        return prior_mean, math.sqrt(prior_var)

    # Recency weighting: exponential decay so recent answers dominate
    # Half-life of ~8 questions: answer from 8 Qs ago has half the weight
    n = len(responses)
    half_life = 8.0
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

    # Target difficulty = current theta (where information is maximized)
    target = session.theta

    # Sort by closeness to target difficulty
    available.sort(key=lambda item: abs(item.difficulty - target))

    # Pick randomly from top 5 closest to add variety
    top_k = min(5, len(available))
    return random.choice(available[:top_k])


def process_answer(session: CATSession, item: Item, chosen: str) -> Response:
    """Process a student's answer and update the session."""
    correct = chosen.upper() == item.answer.upper()
    session._used_ids.add(item.id)

    # Build response history for theta estimation
    history = [(session.items_pool[i].difficulty if i < len(session.items_pool)
                else 0.0, r.correct) for i, r in enumerate(session.responses)]
    # Add current response
    resp_history = []
    for r in session.responses:
        # Find item difficulty from pool
        for it in session.items_pool:
            if it.id == r.item_id:
                resp_history.append((it.difficulty, r.correct))
                break
    resp_history.append((item.difficulty, correct))

    # Update theta
    session.theta, session.se = estimate_theta(resp_history)

    response = Response(
        item_id=item.id,
        chosen=chosen,
        correct=correct,
        theta_after=session.theta,
        se_after=session.se,
    )
    session.responses.append(response)
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
    n = len(session.responses)
    correct = sum(1 for r in session.responses if r.correct)

    # Difficulty breakdown
    tier_stats = {}
    for r in session.responses:
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
        "tier_breakdown": tier_stats,
        "theta_history": [
            {"question": i + 1, "theta": round(r.theta_after, 2), "correct": r.correct}
            for i, r in enumerate(session.responses)
        ],
    }
