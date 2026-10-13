"""Optional Streamlit UI: run streamlit run revenueiq/dashboard.py after training."""
import json
from pathlib import Path
import pandas as pd
import streamlit as st
from revenueiq.data import make_customers, make_demand

st.set_page_config(page_title='RevenueIQ | ML Decision Lab', layout='wide')
st.title('RevenueIQ — ML Decision Lab')
st.warning('Demonstration using synthetic data. No validated business outcomes or production claims.')
metrics_path = Path('artifacts/metrics.json')
if not metrics_path.exists():
    st.info('Train models first: python -m revenueiq.train')
    st.stop()
metrics = json.loads(metrics_path.read_text())
a, b, c, d = st.columns(4)
a.metric('Churn ROC AUC', f"{metrics['churn']['roc_auc']:.3f}")
b.metric('Propensity ROC AUC', f"{metrics['propensity']['roc_auc']:.3f}")
c.metric('CLV MAE', f"{metrics['clv_12m']['mae']:.2f}")
d.metric('Demand MAE', f"{metrics['demand']['mae']:.2f}")
tab1, tab2 = st.tabs(['Customer segments', 'Demand & pricing'])
with tab1:
    df = make_customers()
    st.bar_chart(df.groupby(pd.cut(df.days_since_purchase, bins=[-1, 30, 60, 90, 180], observed=True))['churn'].mean())
    st.dataframe(df.head(20))
with tab2:
    df = make_demand()
    st.line_chart(df.set_index('date')['units'])
    st.caption('Historical synthetic demand; pricing scenarios are served by POST /optimize/price.')
