"""action_feedback.py
Main script wiring together:
- StrokePerception (perception.py)
- StrokeReasoner (reasoning_integration.py)
- Action + feedback loop

Run this file to simulate an end-to-end Perceive -> Reason -> Act -> Feedback cycle.
"""
import time
from typing import List, Dict

import numpy as np

from perception import StrokePerception
from reasoning_integration import StrokeReasoner

def simulate_loop(n_samples: int = 50, noise: bool = True):
    perception = StrokePerception()
    reasoner = StrokeReasoner()

    decisions: List[Dict] = []
    latencies: List[float] = []
    y_true: List[int] = []
    y_pred: List[int] = []

    for i in range(n_samples):
        # Perceive
        idx = np.random.randint(0, len(perception.df))
        sample = perception.get_sample(idx=idx, add_noise=noise)

        # Reason
        result = reasoner.decide(sample)

        # Act
        if result["final_label"] == 1:
            action = "TRIGGER ALERT: High stroke risk patient -> notify clinician."
        else:
            action = "Log as low-risk patient."

        print(f"\nSample {i+1}/{n_samples}")
        print(f"Action: {action}")
        print(f"Details: {result['explanation']}")

        # Collect for evaluation
        gt = perception.get_ground_truth(idx)
        y_true.append(gt)
        y_pred.append(result["final_label"])
        latencies.append(result["latency"])
        decisions.append(result)

    # Feedback & evaluation
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)

    accuracy = (y_true_arr == y_pred_arr).mean()
    avg_latency = float(np.mean(latencies))

    print("\n=== Evaluation Summary ===")
    print(f"Accuracy over {n_samples} samples: {accuracy:.3f}")
    print(f"Average end-to-end latency: {avg_latency*1000:.2f} ms")

    # Simple robustness check: how many noisy samples were still correct?
    robustness = (y_true_arr == y_pred_arr).sum() / len(y_true_arr)
    print(f"Robustness (fraction of correct decisions under noise={noise}): {robustness:.3f}")

    # Feedback: adjust threshold if too many false negatives
    false_negatives = ((y_true_arr == 1) & (y_pred_arr == 0)).sum()
    if false_negatives > 0.3 * (y_true_arr == 1).sum() and (y_true_arr == 1).sum() > 0:
        # Lower threshold slightly
        new_threshold = max(0.3, reasoner.prob_threshold - 0.05)
        print(f"Too many false negatives ({false_negatives}), updating threshold to {new_threshold:.2f}")
        reasoner.update_threshold(new_threshold)
    else:
        print("Threshold kept unchanged.")

    return {
        "accuracy": float(accuracy),
        "avg_latency_ms": avg_latency * 1000.0,
        "robustness": float(robustness),
    }

if __name__ == "__main__":
    simulate_loop(n_samples=30, noise=True)
