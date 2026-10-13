"""Time-safe forecasting and held-out customer model evaluation."""
import json
from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from revenueiq.data import make_customers, make_demand

FEATURES = ['tenure_months', 'monthly_visits', 'avg_order_value', 'discount_rate', 'days_since_purchase']
DEMAND_FEATURES = ['price', 'promo', 'weekday', 'day_of_year']


def train(output='artifacts'):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    customers = make_customers()
    train_idx, test_idx = train_test_split(np.arange(len(customers)), test_size=.25, random_state=42)
    results = {'dataset': 'synthetic', 'customer_holdout_rows': len(test_idx)}
    for target in ['churn', 'propensity', 'clv_12m']:
        classification = target != 'clv_12m'
        model = (HistGradientBoostingClassifier(random_state=42) if classification
                 else HistGradientBoostingRegressor(random_state=42))
        model.fit(customers.iloc[train_idx][FEATURES], customers.iloc[train_idx][target])
        x = customers.iloc[test_idx][FEATURES]
        y = customers.iloc[test_idx][target]
        if classification:
            p = model.predict_proba(x)[:, 1]
            results[target] = {'roc_auc': float(roc_auc_score(y, p)),
                               'average_precision': float(average_precision_score(y, p))}
        else:
            results[target] = {'mae': float(mean_absolute_error(y, model.predict(x)))}
        joblib.dump(model, out / f'{target}.joblib')
    demand = make_demand()
    split = int(len(demand) * .8)
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(demand.iloc[:split][DEMAND_FEATURES], demand.iloc[:split]['units'])
    pred = model.predict(demand.iloc[split:][DEMAND_FEATURES])
    results['demand'] = {'chronological_holdout_rows': len(demand) - split,
                         'mae': float(mean_absolute_error(demand.iloc[split:]['units'], pred))}
    joblib.dump(model, out / 'demand.joblib')
    (out / 'metrics.json').write_text(json.dumps(results, indent=2))
    try:
        import os
        if os.getenv('MLFLOW_TRACKING_URI'):
            import mlflow
            mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])
            mlflow.set_experiment('revenueiq-synthetic')
            with mlflow.start_run():
                mlflow.log_param('dataset', 'synthetic')
                mlflow.log_param('seed', 42)
                for task, metrics in results.items():
                    if isinstance(metrics, dict):
                        for name, value in metrics.items():
                            if isinstance(value, (float, int)):
                                mlflow.log_metric(f'{task}_{name}', value)
                mlflow.log_artifact(str(out / 'metrics.json'))
    except ImportError:
        raise RuntimeError('Install mlflow to enable MLFLOW_TRACKING_URI')
    return results


if __name__ == '__main__':
    print(json.dumps(train(), indent=2))
