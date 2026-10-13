from functools import lru_cache
from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from revenueiq.train import FEATURES, DEMAND_FEATURES

app = FastAPI(title='RevenueIQ', version='0.1.0', description='Synthetic-data ML demonstration; not validated for commercial decisions.')
ARTIFACTS = Path('artifacts')


class Customer(BaseModel):
    tenure_months: int = Field(ge=0)
    monthly_visits: int = Field(ge=0)
    avg_order_value: float = Field(ge=0)
    discount_rate: float = Field(ge=0, le=1)
    days_since_purchase: int = Field(ge=0)


class PricingScenario(BaseModel):
    prices: list[float] = Field(min_length=1, max_length=100)
    promo: int = Field(ge=0, le=1)
    weekday: int = Field(ge=0, le=6)
    day_of_year: int = Field(ge=1, le=366)
    unit_cost: float = Field(ge=0)


@lru_cache(maxsize=4)
def model(name):
    path = ARTIFACTS / f'{name}.joblib'
    if not path.exists():
        raise HTTPException(503, 'Model artifacts missing. Run python -m revenueiq.train first.')
    return joblib.load(path)


@app.get('/health')
def health():
    return {'status': 'ok', 'models_ready': all((ARTIFACTS / f'{n}.joblib').exists() for n in ['churn', 'propensity', 'clv_12m', 'demand'])}


@app.post('/predict/customer')
def predict_customer(customer: Customer):
    row = pd.DataFrame([customer.model_dump()])[FEATURES]
    return {'churn_probability': float(model('churn').predict_proba(row)[0, 1]),
            'purchase_propensity': float(model('propensity').predict_proba(row)[0, 1]),
            'predicted_clv_12m': max(0., float(model('clv_12m').predict(row)[0])),
            'warning': 'Synthetic training data; predictions are illustrative.'}


@app.post('/optimize/price')
def optimize_price(s: PricingScenario):
    if any(p <= 0 for p in s.prices):
        raise HTTPException(422, 'Prices must be positive')
    rows = pd.DataFrame([{'price': p, 'promo': s.promo, 'weekday': s.weekday,
                          'day_of_year': s.day_of_year} for p in s.prices])[DEMAND_FEATURES]
    units = model('demand').predict(rows)
    scenarios = [{'price': p, 'predicted_units': max(0., float(u)),
                  'estimated_margin': float((p - s.unit_cost) * max(0., u))}
                 for p, u in zip(s.prices, units)]
    return {'best': max(scenarios, key=lambda x: x['estimated_margin']), 'scenarios': scenarios,
            'warning': 'Illustrative simulation, not causal price elasticity or a validated optimization policy.'}
