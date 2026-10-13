# RevenueIQ — Customer Intelligence & Pricing ML Platform

**A production-style educational reference project, not a deployed customer production system.** The dataset and outcomes are **synthetic**.

## Scope
- Churn prediction and purchase propensity: gradient-boosted classification; ROC AUC and average precision.
- 12-month CLV prediction: gradient-boosted regression; MAE.
- Demand forecasting: daily demand model with chronological holdout; MAE.
- Pricing: simulated candidate-price margin maximization from predicted demand; **not causal price elasticity**.
- SHAP: optional global feature attribution (`python -m revenueiq.explain`).
- MLflow: optional experiment logging when configured.
- API: FastAPI endpoints `/predict/customer`, `/optimize/price`, `/health`.
- Interactive dashboard: Streamlit, monitoring: PSI drift example, CI: pytest, container: Docker.
- Client-style deliverable: [simulated consulting case](docs/consulting-case.md).

## Run
```bash
python -m pip install -r requirements.txt
python -m revenueiq.train
pytest -q
uvicorn revenueiq.api:app --reload
# separate terminal
streamlit run revenueiq/dashboard.py
```

Visit http://localhost:8000/docs for API documentation. `docker build -t revenueiq . && docker run -p 8000:8000 revenueiq` runs the API.

## Evidence and limitations
The ML models use generated labels and are **not independently validated for real customer decisions**. No customer production outcomes, actual consulting engagements, or live Vertex AI/SageMaker integrations are claimed. Real productionization needs leakage-safe point-in-time features, robust backtesting, baseline comparisons, calibration, security, monitoring and experiments. See [production readiness](docs/production-readiness.md).
