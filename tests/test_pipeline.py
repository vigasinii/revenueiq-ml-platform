from revenueiq.data import make_customers, make_demand
from revenueiq.train import train


def test_reproducible_data():
    assert make_customers(100).equals(make_customers(100))
    assert make_demand(100).equals(make_demand(100))


def test_training(tmp_path):
    metrics = train(tmp_path)
    assert 0 <= metrics['churn']['roc_auc'] <= 1
    assert metrics['demand']['mae'] >= 0
    assert (tmp_path / 'metrics.json').exists()
