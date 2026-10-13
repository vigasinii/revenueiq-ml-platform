"""Reference PSI drift monitor for numeric customer features."""
import numpy as np
from revenueiq.data import make_customers
from revenueiq.train import FEATURES


def population_stability_index(reference, current, bins=10):
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if len(reference) < bins or len(current) == 0:
        raise ValueError('Insufficient observations')
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0 if np.all(current == reference[0]) else float('inf')
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, bins=edges)[0] / len(reference)
    b = np.histogram(current, bins=edges)[0] / len(current)
    a, b = np.clip(a, 1e-6, None), np.clip(b, 1e-6, None)
    return float(np.sum((b - a) * np.log(b / a)))


def demo_report():
    data = make_customers()
    return {f: population_stability_index(data[f].iloc[:2000], data[f].iloc[2000:]) for f in FEATURES}


if __name__ == '__main__':
    print(demo_report())
