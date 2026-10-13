import numpy as np
import pandas as pd


def make_customers(n=3000, seed=42):
    """Synthetic customer-level observations. No real customers or revenue."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 61, n)
    visits = rng.poisson(5, n)
    spend = rng.gamma(3, 35, n).round(2)
    discount = rng.uniform(0, .4, n)
    recency = rng.integers(0, 180, n)
    logits = -1.8 + .014 * recency - .025 * tenure - .13 * visits + 1.2 * discount
    churn_prob = 1 / (1 + np.exp(-logits))
    churn = rng.binomial(1, churn_prob)
    propensity = rng.binomial(1, 1 / (1 + np.exp(-(-1 + .2 * visits - .012 * recency + .006 * spend))))
    clv = np.maximum(0, spend * (1 + .12 * visits) * (1 - .55 * churn) + rng.normal(0, 30, n))
    return pd.DataFrame(dict(tenure_months=tenure, monthly_visits=visits, avg_order_value=spend,
                             discount_rate=discount, days_since_purchase=recency,
                             churn=churn, propensity=propensity, clv_12m=clv.round(2)))


def make_demand(n=730, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2023-01-01', periods=n, freq='D')
    price = rng.uniform(8, 16, n)
    promo = rng.binomial(1, .15, n)
    weekday = dates.dayofweek.to_numpy()
    seasonal = 12 * np.sin(np.arange(n) * 2 * np.pi / 365)
    demand = np.maximum(0, 110 - 5 * price + 16 * promo + 8 * (weekday >= 5) + seasonal + rng.normal(0, 5, n))
    return pd.DataFrame(dict(date=dates, price=price.round(2), promo=promo, weekday=weekday,
                             day_of_year=dates.dayofyear, units=np.rint(demand).astype(int)))
