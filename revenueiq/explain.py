"""Optional SHAP explanations for fitted tree-based estimators."""
import json
from pathlib import Path
import joblib
import numpy as np
from revenueiq.data import make_customers, make_demand
from revenueiq.train import FEATURES, DEMAND_FEATURES


def explain(artifact_dir='artifacts', sample_size=100):
    import shap
    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for target in ('churn', 'propensity', 'clv_12m', 'demand'):
        model = joblib.load(out / f'{target}.joblib')
        features = DEMAND_FEATURES if target == 'demand' else FEATURES
        dataset = make_demand() if target == 'demand' else make_customers()
        background = dataset[features].iloc[:100]
        sample = dataset[features].iloc[-sample_size:]
        explainer = shap.Explainer(model.predict if target in ('clv_12m', 'demand') else lambda x: model.predict_proba(x)[:, 1], background)
        values = explainer(sample).values
        results[target] = dict(zip(features, np.abs(values).mean(axis=0).astype(float).tolist()))
    (out / 'shap_importance.json').write_text(json.dumps(results, indent=2))
    return results


if __name__ == '__main__':
    print(json.dumps(explain(), indent=2))
