"""
llm_correctness_demo.py
=======================

Companion code for the blog post "Knowing When Your LLM Is Wrong."

A single, self-contained script that walks through the full evaluation
pipeline for a binary routing agent:

    1. Build a labeled gold set
    2. Run the agent and compute error rate + confusion matrix
    3. Estimate the Bayes floor from inter-annotator agreement
    4. Extract a confidence signal via self-consistency sampling
    5. Calibrate the signal with Platt scaling
    6. Measure calibration with ECE and a reliability diagram
    7. Tune a decision threshold (with an abstention zone)
    8. Run an A/B test between two policies, properly

The "LLM" here is a mock function so the script runs offline with no
API key. The structure is identical to what you'd write against a real
commercial API: replace `MockLLM.classify` with a real API call and the
rest of the pipeline is unchanged.

Dependencies: numpy, scikit-learn, matplotlib
    pip install numpy scikit-learn matplotlib

Run:
    python llm_correctness_demo.py
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# 0. Setup: a mock routing problem
# ---------------------------------------------------------------------------
#
# task_1 = "answer from knowledge"   (label 0)
# task_2 = "search the web"          (label 1)
#
# We pretend each user query has a true intent. A real system would replace
# this with real labeled data sampled from production traffic.

random.seed(42)
np.random.seed(42)

EXAMPLES: list[tuple[str, int]] = [
    # (query, true_label)
    ("what is the capital of France",                 0),
    ("who wrote Hamlet",                              0),
    ("define photosynthesis",                         0),
    ("explain the Pythagorean theorem",               0),
    ("what are the laws of thermodynamics",           0),
    ("when did World War II end",                     0),
    ("what is the speed of light",                    0),
    ("what's the weather in Paris right now",         1),
    ("latest news on the Fed rate decision",          1),
    ("current stock price of Apple",                  1),
    ("who won the game last night",                   1),
    ("any updates on the SpaceX launch today",        1),
    ("what's trending on Twitter",                    1),
    ("recent earthquake reports",                     1),
    # genuinely ambiguous cases (Bayes-error territory)
    ("tell me about cats",                            0),  # could go either way
    ("how is the market doing",                       1),  # ambiguous: definition vs current state
]


# ---------------------------------------------------------------------------
# 1. The "agent": a mock LLM-based router
# ---------------------------------------------------------------------------
#
# Replace this class with real API calls in production. The rest of the
# pipeline doesn't care how the decision is made.

class MockLLM:
    """A mock LLM-based router with tunable accuracy and miscalibration.

    Returns a probability over labels (0, 1). Genuinely ambiguous queries
    get probabilities near 0.5 (aleatoric uncertainty). Easy queries get
    extreme probabilities. We deliberately make it overconfident, like
    real LLMs, so calibration has something to fix.
    """

    def __init__(self, accuracy: float = 0.85, overconfidence: float = 1.6):
        self.accuracy = accuracy
        self.overconfidence = overconfidence  # > 1 makes outputs more extreme

    def _underlying_prob(self, query: str, true_label: int) -> float:
        """Probability the LLM puts on label=1, given the query."""
        # Ambiguous queries straddle 0.5
        if "cats" in query or "market" in query:
            return random.uniform(0.35, 0.65)
        # Otherwise, lean toward the truth most of the time
        if random.random() < self.accuracy:
            base = random.uniform(0.7, 0.95) if true_label == 1 else random.uniform(0.05, 0.30)
        else:
            base = random.uniform(0.55, 0.75) if true_label == 0 else random.uniform(0.25, 0.45)
        return base

    def classify(self, query: str, true_label: int, temperature: float = 1.0) -> int:
        """Single deterministic-ish classification. Returns 0 or 1."""
        p = self._underlying_prob(query, true_label)
        # Apply overconfidence: push probabilities toward extremes
        p_sharp = self._sharpen(p, self.overconfidence)
        # Sample if temperature > 0 (mimics non-zero-temperature decoding)
        if temperature > 0:
            return int(random.random() < p_sharp)
        return int(p_sharp >= 0.5)

    def classify_with_confidence(self, query: str, true_label: int) -> tuple[int, float]:
        """Returns (prediction, raw_confidence_in_prediction)."""
        p = self._underlying_prob(query, true_label)
        p_sharp = self._sharpen(p, self.overconfidence)
        prediction = int(p_sharp >= 0.5)
        confidence = p_sharp if prediction == 1 else (1 - p_sharp)
        return prediction, confidence

    @staticmethod
    def _sharpen(p: float, factor: float) -> float:
        """Push p away from 0.5 by `factor` in logit space (overconfidence)."""
        eps = 1e-6
        p = min(max(p, eps), 1 - eps)
        logit = math.log(p / (1 - p))
        return 1 / (1 + math.exp(-logit * factor))


# ---------------------------------------------------------------------------
# 2. Error rate, confusion matrix, confidence intervals
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    n: int
    accuracy: float
    error_rate: float
    ci_low: float
    ci_high: float
    tp: int
    fp: int
    tn: int
    fn: int


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def evaluate(predictions: list[int], labels: list[int]) -> EvalResult:
    n = len(labels)
    correct = sum(p == y for p, y in zip(predictions, labels))
    accuracy = correct / n
    err = 1 - accuracy

    # Confusion matrix (positive class = 1 = task_2)
    tp = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 1)
    fp = sum(1 for p, y in zip(predictions, labels) if p == 1 and y == 0)
    tn = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 0)
    fn = sum(1 for p, y in zip(predictions, labels) if p == 0 and y == 1)

    # Wilson interval on the error rate
    lo, hi = wilson_interval(n - correct, n)
    return EvalResult(n, accuracy, err, lo, hi, tp, fp, tn, fn)


def estimate_bayes_floor(labelers: list[list[int]]) -> float:
    """Estimate the irreducible error from inter-annotator disagreement.

    Given multiple labeler arrays (same items, same order), returns the
    fraction of items where labelers disagreed. This is a lower bound
    on what any agent can achieve.
    """
    n_items = len(labelers[0])
    disagreements = 0
    for i in range(n_items):
        votes = [lab[i] for lab in labelers]
        if len(set(votes)) > 1:
            disagreements += 1
    return disagreements / n_items


# ---------------------------------------------------------------------------
# 3. Self-consistency: extract confidence by sampling
# ---------------------------------------------------------------------------

def self_consistency_confidence(
    llm: MockLLM, query: str, true_label: int, n_samples: int = 20
) -> tuple[int, float]:
    """Run the LLM N times, return majority prediction and vote fraction."""
    votes = [llm.classify(query, true_label, temperature=1.0) for _ in range(n_samples)]
    pred = 1 if sum(votes) > n_samples / 2 else 0
    # Confidence = fraction of votes for the predicted class
    confidence = votes.count(pred) / n_samples
    return pred, confidence


# ---------------------------------------------------------------------------
# 4. Calibration: Platt scaling + ECE + reliability diagram
# ---------------------------------------------------------------------------

class PlattCalibrator:
    """Logistic regression on raw confidence -> calibrated probability."""

    def __init__(self):
        self.model = LogisticRegression()

    def fit(self, raw_confidences: np.ndarray, correct: np.ndarray) -> "PlattCalibrator":
        # We fit p(correct=1 | raw_confidence)
        X = raw_confidences.reshape(-1, 1)
        self.model.fit(X, correct)
        return self

    def transform(self, raw_confidences: np.ndarray) -> np.ndarray:
        X = raw_confidences.reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> float:
    """ECE: weighted average of |accuracy - confidence| per confidence bin."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for i in range(n_bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:  # include 1.0 in the last bin
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        bin_count = in_bin.sum()
        if bin_count == 0:
            continue
        bin_acc = correct[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (bin_count / n) * abs(bin_acc - bin_conf)
    return ece


def reliability_data(
    confidences: np.ndarray, correct: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (bin_centers, bin_accuracy, bin_count) for plotting."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    centers, accs, counts = [], [], []
    for i in range(n_bins):
        in_bin = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
        if i == n_bins - 1:
            in_bin = (confidences >= bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if in_bin.sum() == 0:
            continue
        centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        accs.append(correct[in_bin].mean())
        counts.append(in_bin.sum())
    return np.array(centers), np.array(accs), np.array(counts)


# ---------------------------------------------------------------------------
# 5. Decision threshold with abstention zone
# ---------------------------------------------------------------------------

def route_with_abstention(
    calibrated_prob: float, low: float = 0.35, high: float = 0.65
) -> str:
    """Map a calibrated P(label=1) to an action.

    The width of the abstention zone (high - low) is a product decision:
    wider = more cautious (more escalations, fewer mistakes), narrower =
    more decisive (fewer escalations, more mistakes). Tune on a held-out
    set against your cost asymmetry.
    """
    if calibrated_prob < low:
        return "task_1"
    if calibrated_prob > high:
        return "task_2"
    return "abstain"  # escalate, ask clarifying question, or run both


# ---------------------------------------------------------------------------
# 6. A/B test: two-proportion z-test with Wilson CI on the difference
# ---------------------------------------------------------------------------

def two_proportion_test(
    successes_a: int, n_a: int, successes_b: int, n_b: int
) -> dict:
    """Two-proportion z-test plus a normal-approx CI on the difference."""
    p_a = successes_a / n_a
    p_b = successes_b / n_b
    diff = p_b - p_a

    # Pooled variance for the test
    p_pool = (successes_a + successes_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0

    # Unpooled SE for the CI
    se_unpooled = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci_low = diff - 1.96 * se_unpooled
    ci_high = diff + 1.96 * se_unpooled

    # Two-sided p-value (normal approx)
    p_value = 2 * (1 - _phi(abs(z)))
    return {
        "p_a": p_a,
        "p_b": p_b,
        "diff": diff,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "z": z,
        "p_value": p_value,
    }


def required_sample_size(p_bar: float, mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Per-arm sample size for two-proportion z-test, normal approx."""
    z_alpha = 1.96 if alpha == 0.05 else 2.576
    z_beta = 0.84 if power == 0.8 else 1.28
    return math.ceil(2 * p_bar * (1 - p_bar) * (z_alpha + z_beta) ** 2 / mde**2)


def _phi(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


# ---------------------------------------------------------------------------
# 7. The full pipeline, glued together
# ---------------------------------------------------------------------------

def run_demo() -> None:
    print("=" * 68)
    print("LLM correctness pipeline demo")
    print("=" * 68)

    # We expand the small dataset by replicating with noise to get useful
    # sample sizes for statistics. In real life you sample from production.
    queries_labels = []
    for _ in range(60):
        queries_labels.extend(EXAMPLES)
    queries = [q for q, _ in queries_labels]
    labels = [y for _, y in queries_labels]
    n = len(labels)
    print(f"\nGold-set size: {n} examples")

    # --- Step 1: baseline evaluation -----------------------------------
    llm_a = MockLLM(accuracy=0.82, overconfidence=1.8)
    preds_a = [llm_a.classify(q, y, temperature=0.0) for q, y in zip(queries, labels)]
    res_a = evaluate(preds_a, labels)

    print("\n--- 1. Baseline error rate (policy A) ---")
    print(f"Error rate:  {res_a.error_rate:.3f}  (95% CI [{res_a.ci_low:.3f}, {res_a.ci_high:.3f}])")
    print("Confusion matrix (positive class = task_2):")
    print(f"               pred=task_1   pred=task_2")
    print(f"  true=task_1  TN={res_a.tn:<6}    FP={res_a.fp:<6}")
    print(f"  true=task_2  FN={res_a.fn:<6}    TP={res_a.tp:<6}")

    # --- Step 2: Bayes floor estimate ----------------------------------
    # Simulate three independent human labelers
    def noisy_labeler(true_label: int, query: str) -> int:
        if "cats" in query or "market" in query:
            return random.randint(0, 1)  # genuinely ambiguous
        return true_label if random.random() > 0.02 else 1 - true_label

    labelers = [
        [noisy_labeler(y, q) for q, y in zip(queries, labels)] for _ in range(3)
    ]
    bayes_floor = estimate_bayes_floor(labelers)
    print(f"\n--- 2. Bayes floor (inter-annotator disagreement) ---")
    print(f"Estimated irreducible error: {bayes_floor:.3f}")
    print("(No agent can do better than this on this distribution.)")

    # --- Step 3: extract confidence via self-consistency --------------
    print("\n--- 3. Confidence via self-consistency (10 samples) ---")
    # raw_p1[i] = fraction of samples voting for label=1
    raw_p1 = []
    sc_preds = []
    for q, y in zip(queries, labels):
        votes = [llm_a.classify(q, y, temperature=1.0) for _ in range(10)]
        p1 = sum(votes) / len(votes)
        raw_p1.append(p1)
        sc_preds.append(int(p1 >= 0.5))
    raw_p1 = np.array(raw_p1)
    # confidence in own prediction = max(p1, 1 - p1)
    raw_confidences = np.maximum(raw_p1, 1 - raw_p1)
    correct = np.array([int(p == y) for p, y in zip(sc_preds, labels)])
    print(f"Mean raw confidence: {raw_confidences.mean():.3f}")
    print(f"Mean accuracy:       {correct.mean():.3f}")
    print("(Gap between the two = miscalibration to fix.)")

    # --- Step 4: calibrate with Platt scaling --------------------------
    # We calibrate P(label=1 | raw_p1) directly: fit logistic regression
    # mapping the raw vote fraction to the true label.
    idx = np.arange(n)
    np.random.shuffle(idx)
    cal_idx, eval_idx = idx[: n // 2], idx[n // 2 :]

    calibrator = PlattCalibrator()
    calibrator.fit(raw_p1[cal_idx], np.array(labels)[cal_idx])
    calibrated_p1 = calibrator.transform(raw_p1[eval_idx])

    # For ECE, compare confidence-in-prediction to actual correctness
    eval_correct = correct[eval_idx]
    raw_conf_eval = raw_confidences[eval_idx]
    cal_conf = np.maximum(calibrated_p1, 1 - calibrated_p1)

    ece_raw = expected_calibration_error(raw_conf_eval, eval_correct)
    ece_cal = expected_calibration_error(cal_conf, eval_correct)

    print("\n--- 4. Calibration with Platt scaling ---")
    print(f"ECE before:  {ece_raw:.4f}")
    print(f"ECE after:   {ece_cal:.4f}")
    if ece_raw > 0:
        print(f"Reduction:   {(1 - ece_cal / ece_raw) * 100:.1f}%")

    # --- Step 5: routing with abstention zone --------------------------
    print("\n--- 5. Routing decisions with abstention zone (0.35 < p < 0.65) ---")
    actions = [route_with_abstention(p) for p in calibrated_p1]
    n_abstain = actions.count("abstain")
    n_t1 = actions.count("task_1")
    n_t2 = actions.count("task_2")
    print(f"Routed to task_1: {n_t1}")
    print(f"Routed to task_2: {n_t2}")
    print(f"Abstained:        {n_abstain}  ({n_abstain / len(actions) * 100:.1f}%)")

    # --- Step 6: A/B test against a "better" policy --------------------
    llm_b = MockLLM(accuracy=0.88, overconfidence=1.4)  # genuinely better
    preds_b = [llm_b.classify(q, y, temperature=0.0) for q, y in zip(queries, labels)]
    res_b = evaluate(preds_b, labels)

    successes_a = res_a.n - sum(1 for p, y in zip(preds_a, labels) if p != y)
    successes_b = res_b.n - sum(1 for p, y in zip(preds_b, labels) if p != y)
    test = two_proportion_test(successes_a, res_a.n, successes_b, res_b.n)

    print("\n--- 6. A/B test: policy A vs policy B ---")
    print(f"Accuracy A: {test['p_a']:.3f}")
    print(f"Accuracy B: {test['p_b']:.3f}")
    print(f"Diff (B-A): {test['diff']:+.3f}  (95% CI [{test['ci_low']:+.3f}, {test['ci_high']:+.3f}])")
    print(f"Two-sided p-value: {test['p_value']:.4f}")

    # Power calculation: how big a sample would we have needed?
    p_bar = (test["p_a"] + test["p_b"]) / 2
    needed = required_sample_size(1 - p_bar, mde=0.02)
    print(f"\nFor MDE=2pp at 80% power, you'd need ~{needed} per arm.")

    # --- Step 7: plots --------------------------------------------------
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

        # Reliability diagram
        c_raw, a_raw, _ = reliability_data(raw_conf_eval, eval_correct)
        c_cal, a_cal, _ = reliability_data(cal_conf, eval_correct)
        axes[0].plot([0, 1], [0, 1], "--", color="gray", label="perfect calibration")
        axes[0].plot(c_raw, a_raw, "o-", label=f"raw (ECE={ece_raw:.3f})")
        axes[0].plot(c_cal, a_cal, "s-", label=f"calibrated (ECE={ece_cal:.3f})")
        axes[0].set_xlabel("predicted confidence")
        axes[0].set_ylabel("observed accuracy")
        axes[0].set_title("Reliability diagram")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # Confusion matrix as a heatmap
        cm = np.array([[res_a.tn, res_a.fp], [res_a.fn, res_a.tp]])
        im = axes[1].imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                             color="white" if cm[i, j] > cm.max() / 2 else "black",
                             fontsize=14)
        axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["pred task_1", "pred task_2"])
        axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["true task_1", "true task_2"])
        axes[1].set_title(f"Confusion matrix (policy A, n={res_a.n})")
        plt.colorbar(im, ax=axes[1])

        plt.tight_layout()
        plt.savefig("calibration_and_confusion.png", dpi=120)
        print("\nSaved plot: calibration_and_confusion.png")
    except ImportError:
        print("\n(matplotlib not installed; skipping plots)")


if __name__ == "__main__":
    run_demo()
