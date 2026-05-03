"""
FDA Drug Approval & Stock Movement Predictor
Research by Hari Vykuntapu | MS AI, Southwest Baptist University
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Resolve paths relative to this file so they work both locally and on Streamlit Cloud
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))


def _path(*parts):
    return os.path.join(_ROOT, *parts)

st.set_page_config(
    page_title="FDA Drug Approval & Stock Movement Predictor",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CSS ---
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1a237e;
        line-height: 1.2;
    }
    .sub-header {
        font-size: 1rem;
        color: #5c6bc0;
        margin-bottom: 1.5rem;
    }
    .rss-formula {
        background: linear-gradient(135deg, #e8eaf6, #f3e5f5);
        border-left: 4px solid #3f51b5;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    .footer-text {
        font-size: 0.8rem;
        color: #9e9e9e;
        text-align: center;
        padding-top: 1rem;
        border-top: 1px solid #e0e0e0;
    }
    .highlight-box {
        background: #e3f2fd;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# --- Utilities ---
@st.cache_data
def load_features():
    path = _path("data", "processed", "fda_features.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["approval_date"])
    return None


@st.cache_data
def load_metrics():
    path = _path("outputs", "results", "model_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None


@st.cache_resource
def load_model():
    import joblib
    path = _path("models", "best_model.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_data
def load_metadata():
    path = _path("models", "model_metadata.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def compute_rss(approval_type_weight, drug_novelty_score, market_timing_factor):
    return round(
        approval_type_weight * 0.45 +
        drug_novelty_score * 0.35 +
        market_timing_factor * 0.20,
        4
    )


# --- Sidebar ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/63/Wikipedia-logo-v2-en.svg/1200px-Wikipedia-logo-v2-en.svg.png",
             width=50)
    st.markdown("### Navigation")
    tab_selection = st.radio(
        "Select Tab",
        ["🏠 HOME", "🔮 PREDICT", "📊 RESEARCH", "👤 ABOUT"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("**Project Links**")
    st.markdown("[GitHub Repository](https://github.com/HariVykuntapu/fda-stock-prediction)")
    st.markdown("[LinkedIn](https://www.linkedin.com/in/harivykuntapu)")

    st.markdown("---")
    st.markdown('<div class="footer-text">© 2026 Hari Vykuntapu</div>', unsafe_allow_html=True)


# ============================================================
# TAB: HOME
# ============================================================
if "HOME" in tab_selection:
    st.markdown('<div class="main-header">💊 FDA Drug Approval & Stock Movement Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Research by Hari Vykuntapu | MS Artificial Intelligence, Southwest Baptist University, United States</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    _card = (
        "background:#f0f4ff;border-radius:10px;padding:1.1rem 1rem;"
        "text-align:center;border:1px solid #c5cae9;"
    )
    _h = "margin:0 0 0.3rem 0;font-size:1.1rem;font-weight:700;color:#1a237e;"
    _p = "margin:0;font-size:0.85rem;color:#555;"
    with col1:
        st.markdown(
            f'<div style="{_card}"><div style="{_h}">296 Drug Approvals Analyzed</div>'
            f'<div style="{_p}">2018 – 2023 · 12 pharma companies</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div style="{_card}"><div style="{_h}">XGBoost AUROC: 0.5725</div>'
            f'<div style="{_p}">Best model · above random baseline</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div style="{_card}"><div style="{_h}">FinBERT + RSS</div>'
            f'<div style="{_p}">Novel feature engineering · top SHAP features</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        st.markdown("### Why I Built This")
        st.markdown("""
        I kept noticing something: every FDA drug approval hits the news in the same format, but some move the
        stock 10% and others barely register. The announcement text is nearly identical. What varies is the
        *regulatory details* — what kind of drug it is, how novel, what classification the FDA used.

        Most stock prediction work optimizes RSI and MACD. I wanted to know if the **structure of the regulatory
        decision itself** — not the words, but the classification metadata — carries a signal that technical
        indicators miss entirely. So I pulled six years of FDA approvals, matched them to stock prices, and built
        a feature to test it.
        """)

        st.markdown("### The Regulatory Sentiment Score (RSS)")
        st.markdown("*Original contribution by Hari Vykuntapu*")
        st.markdown(
            """
            <div style="background:linear-gradient(135deg,#e8eaf6,#f3e5f5);
                        border-left:4px solid #3f51b5;border-radius:8px;
                        padding:1rem 1.4rem;font-family:'Courier New',monospace;
                        font-size:0.92rem;line-height:1.7;">
            <strong>RSS = (approval_type_weight &times; 0.45)
                  + (drug_novelty_score &times; 0.35)
                  + (market_timing_factor &times; 0.20)</strong><br><br>
            Where:<br>
            &nbsp;&nbsp;&bull;&nbsp; approval_type_weight: NDA&nbsp;=&nbsp;1.0 | BLA&nbsp;=&nbsp;0.8 | ANDA&nbsp;=&nbsp;0.3<br>
            &nbsp;&nbsp;&bull;&nbsp; drug_novelty_score: First-in-class&nbsp;=&nbsp;1.0 | Follow-on&nbsp;=&nbsp;0.5 | Generic&nbsp;=&nbsp;0.0<br>
            &nbsp;&nbsp;&bull;&nbsp; market_timing_factor: day-of-week &times; month-end &times; Q4 (range 0&ndash;1)
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:
        st.markdown("### Companies Studied")
        companies = {
            'Ticker': ['PFE', 'MRNA', 'JNJ', 'AZN', 'MRK', 'BMY', 'LLY', 'ABBV', 'GILD', 'AMGN', 'BIIB', 'REGN'],
            'Company': ['Pfizer', 'Moderna', 'Johnson & Johnson', 'AstraZeneca', 'Merck',
                        'Bristol-Myers Squibb', 'Eli Lilly', 'AbbVie', 'Gilead Sciences',
                        'Amgen', 'Biogen', 'Regeneron']
        }
        st.dataframe(pd.DataFrame(companies), hide_index=True, height=380)

    st.markdown("---")

    # Load data for preview
    df = load_features()
    if df is not None:
        st.markdown("### Recent Approval Events (Preview)")
        preview_cols = ['ticker', 'drug_name', 'approval_date', 'app_type_clean', 'rss_score', 'return_7d', 'price_up_7d']
        preview_cols = [c for c in preview_cols if c in df.columns]
        st.dataframe(
            df[preview_cols].sort_values('approval_date', ascending=False).head(10),
            hide_index=True
        )


# ============================================================
# TAB: PREDICT
# ============================================================
elif "PREDICT" in tab_selection:
    st.markdown('<div class="main-header">🔮 Predict Stock Movement</div>', unsafe_allow_html=True)
    st.markdown(
        "Give me the regulatory details of an FDA approval — application type, drug novelty class, company, date. "
        "I'll compute the RSS, score the FinBERT sentiment, and run it through the trained XGBoost to give you "
        "a 7-day directional prediction and probability split. "
        "No model here will tell you what to trade. But it will show you what the regulatory structure implies."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Regulatory Inputs")

        application_type = st.selectbox(
            "Application Type",
            options=["NDA — New Drug Application", "BLA — Biologics License Application", "ANDA — Generic Drug"],
            help="NDA and BLA cover brand-name drugs. ANDA covers generics."
        )

        drug_class = st.selectbox(
            "Drug Novelty Class",
            options=[
                "Type 1 — New Molecular Entity (First-in-Class)",
                "Type 2 — New Active Ingredient",
                "Type 3 — New Dosage Form (Follow-on)",
                "Type 4 — New Combination (Follow-on)",
                "Type 5 — New Formulation (Follow-on)",
                "Type 6 — New Indication (Follow-on)",
                "Generic Equivalent",
            ]
        )

        company = st.selectbox(
            "Company",
            options=["Pfizer (PFE)", "Moderna (MRNA)", "Johnson & Johnson (JNJ)", "AstraZeneca (AZN)",
                     "Merck (MRK)", "Bristol-Myers Squibb (BMY)", "Eli Lilly (LLY)", "AbbVie (ABBV)",
                     "Gilead Sciences (GILD)", "Amgen (AMGN)", "Biogen (BIIB)", "Regeneron (REGN)"]
        )

    with col2:
        st.markdown("#### Market Timing Inputs")

        approval_date = st.date_input("Approval Date", value=pd.Timestamp("2024-01-15"))

        finbert_score = st.slider(
            "FinBERT Sentiment Score",
            min_value=-1.0, max_value=1.0, value=0.25, step=0.05,
            help="Finance-domain NLP sentiment. Positive = bullish language, Negative = cautionary."
        )

        vader_score = st.slider(
            "VADER Sentiment Score",
            min_value=-1.0, max_value=1.0, value=0.15, step=0.05,
            help="General-purpose lexicon sentiment score."
        )

    # Derive features
    app_map = {"NDA — New Drug Application": "NDA", "BLA — Biologics License Application": "BLA", "ANDA — Generic Drug": "ANDA"}
    app_type = app_map.get(application_type, "NDA")
    type_weights = {"NDA": 1.0, "BLA": 0.8, "ANDA": 0.3}
    approval_type_weight = type_weights[app_type]

    if "Generic" in drug_class or "ANDA" in app_type:
        drug_novelty_score = 0.0
    elif "First-in-Class" in drug_class or "Type 1" in drug_class or "Type 2" in drug_class:
        drug_novelty_score = 1.0
    else:
        drug_novelty_score = 0.5

    day_weights = {0: 0.9, 1: 1.0, 2: 0.95, 3: 0.9, 4: 0.7}
    dow = pd.Timestamp(str(approval_date)).dayofweek
    day_factor = day_weights.get(dow, 0.8)
    month_factor = 0.85 if approval_date.day >= 28 else 1.0
    q4_factor = 1.1 if approval_date.month in [10, 11] else 1.0
    market_timing_factor = min(day_factor * month_factor * q4_factor, 1.0)

    rss_score = compute_rss(approval_type_weight, drug_novelty_score, market_timing_factor)

    ticker_map = {"Pfizer (PFE)": 3, "Moderna (MRNA)": 1, "Johnson & Johnson (JNJ)": 3,
                  "AstraZeneca (AZN)": 2, "Merck (MRK)": 3, "Bristol-Myers Squibb (BMY)": 2,
                  "Eli Lilly (LLY)": 3, "AbbVie (ABBV)": 3, "Gilead Sciences (GILD)": 2,
                  "Amgen (AMGN)": 2, "Biogen (BIIB)": 2, "Regeneron (REGN)": 2}
    market_cap_cat = ticker_map.get(company, 2)

    st.markdown("---")
    st.markdown("#### Computed RSS Components")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approval Type Weight", f"{approval_type_weight:.2f}", f"Type: {app_type}")
    c2.metric("Drug Novelty Score", f"{drug_novelty_score:.2f}")
    c3.metric("Market Timing Factor", f"{market_timing_factor:.3f}", f"Day: {['Mon','Tue','Wed','Thu','Fri'][min(dow,4)]}")
    c4.metric("📊 RSS Score", f"{rss_score:.4f}", delta="Novel = higher RSS" if rss_score > 0.6 else "Lower novelty signal")

    # RSS gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=rss_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Regulatory Sentiment Score (RSS)", 'font': {'size': 18}},
        delta={'reference': 0.6, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
        gauge={
            'axis': {'range': [0, 1.1], 'tickwidth': 1},
            'bar': {'color': "#3f51b5"},
            'steps': [
                {'range': [0, 0.35], 'color': '#ffcdd2'},
                {'range': [0.35, 0.65], 'color': '#fff9c4'},
                {'range': [0.65, 1.1], 'color': '#c8e6c9'}
            ],
            'threshold': {'line': {'color': "black", 'width': 3}, 'thickness': 0.8, 'value': rss_score}
        }
    ))
    fig_gauge.update_layout(height=280, margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Prediction
    st.markdown("---")
    st.markdown("#### Model Prediction")

    model = load_model()
    metadata = load_metadata()

    if model is not None and metadata is not None:
        feature_cols = metadata.get('feature_cols', [])
        input_dict = {
            'rss_score': rss_score,
            'approval_type_weight': approval_type_weight,
            'drug_novelty_score': drug_novelty_score,
            'market_timing_factor': market_timing_factor,
            'finbert_compound': finbert_score,
            'vader_compound': vader_score,
            'market_cap_category': market_cap_cat,
            'day_of_week': dow,
            'approval_quarter': pd.Timestamp(str(approval_date)).quarter,
            'prior_approvals_count': 5,
        }

        input_row = pd.DataFrame([[input_dict.get(c, 0.5) for c in feature_cols]], columns=feature_cols)

        try:
            pred = model.predict(input_row)[0]
            prob = model.predict_proba(input_row)[0]
            prob_up = prob[1]
            prob_down = prob[0]

            pred_col1, pred_col2 = st.columns(2)
            if pred == 1:
                pred_col1.success(f"📈 **STOCK LIKELY UP** in 7 days\n\nConfidence: {prob_up:.1%}")
            else:
                pred_col1.error(f"📉 **STOCK LIKELY DOWN** in 7 days\n\nConfidence: {prob_down:.1%}")

            fig_prob = go.Figure(go.Bar(
                x=['Probability DOWN', 'Probability UP'],
                y=[prob_down, prob_up],
                marker_color=['#ef5350', '#66bb6a'],
                text=[f'{prob_down:.1%}', f'{prob_up:.1%}'],
                textposition='outside'
            ))
            fig_prob.update_layout(
                title='Prediction Probability',
                yaxis_range=[0, 1.1],
                height=320,
                showlegend=False,
                margin=dict(t=50, b=20, l=20, r=20)
            )
            pred_col2.plotly_chart(fig_prob, use_container_width=True)

        except Exception as e:
            st.warning(f"Prediction error: {e}. Run the model training notebook first.")
    else:
        st.info("No trained model found. Run `notebooks/04_model_training.ipynb` first to train and save the model.")

        # Show what RSS score implies based on EDA patterns
        if rss_score > 0.7:
            st.success(f"RSS = {rss_score:.3f} — High regulatory signal. Based on EDA patterns, high-RSS approvals have historically shown higher probability of positive 7-day returns.")
        elif rss_score > 0.45:
            st.warning(f"RSS = {rss_score:.3f} — Moderate regulatory signal. Mixed historical outcomes.")
        else:
            st.error(f"RSS = {rss_score:.3f} — Low regulatory signal. Generic or follow-on approvals tend to produce minimal market reaction.")

    st.markdown("---")
    st.caption("⚠️ This is a research tool, not financial advice. Stock prices are influenced by many factors beyond FDA approvals.")


# ============================================================
# TAB: RESEARCH
# ============================================================
elif "RESEARCH" in tab_selection:
    st.markdown('<div class="main-header">📊 Research Results</div>', unsafe_allow_html=True)
    st.markdown(
        "XGBoost hit AUROC 0.5725 on the held-out test set — above random, not enough to trade on. "
        "The more interesting result is the SHAP breakdown: FinBERT compound sentiment ranked first, "
        "approval type weight second, RSS third. I built this expecting structured metadata to win. "
        "What I found is that they're both contributing, and they're measuring different things."
    )
    st.markdown("---")

    # Load data
    df = load_features()
    metrics = load_metrics()

    # Metrics table — loaded from CSV if available, otherwise hardcoded from training run
    st.markdown("### Model Performance")
    _hardcoded_metrics = pd.DataFrame({
        "model":    ["Logistic Regression", "Random Forest", "XGBoost"],
        "accuracy": [0.5333, 0.5000, 0.5333],
        "precision":[0.5455, 0.5263, 0.5625],
        "recall":   [0.7500, 0.6250, 0.5625],
        "f1":       [0.6316, 0.5714, 0.5625],
        "auroc":    [0.4431, 0.5525, 0.5725],
    }).set_index("model")

    if metrics is not None:
        display_metrics = metrics
    else:
        display_metrics = _hardcoded_metrics

    st.dataframe(
        display_metrics.style.highlight_max(axis=0, color="#c8e6c9").format("{:.4f}"),
        use_container_width=True,
    )

    st.markdown("""
    **Best model:** XGBoost — AUROC 0.5725, 236 train / 60 test samples

    **Top SHAP features** (mean |SHAP| on test set):
    1. `finbert_compound` — FinBERT finance-domain sentiment
    2. `approval_type_weight` — NDA / BLA / ANDA tier (core RSS component)
    3. `rss_score` — composite regulatory signal (ranks above individual components)
    """)

    st.markdown("---")

    if df is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Approval Type Distribution")
            app_counts = df['app_type_clean'].value_counts() if 'app_type_clean' in df.columns else pd.Series()
            if not app_counts.empty:
                fig_pie = px.pie(
                    values=app_counts.values,
                    names=app_counts.index,
                    color_discrete_map={'NDA': '#2196F3', 'BLA': '#4CAF50', 'ANDA': '#FF9800', 'Other': '#9C27B0'},
                    hole=0.4
                )
                fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                fig_pie.update_layout(height=340, margin=dict(t=20, b=20))
                st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            st.markdown("### 7-Day Return by Approval Type")
            if 'app_type_clean' in df.columns and 'return_7d' in df.columns:
                type_ret = df.groupby('app_type_clean')['return_7d'].agg(['mean', 'sem']).reset_index()
                fig_bar = px.bar(
                    type_ret,
                    x='app_type_clean',
                    y='mean',
                    error_y='sem',
                    color='app_type_clean',
                    color_discrete_map={'NDA': '#2196F3', 'BLA': '#4CAF50', 'ANDA': '#FF9800', 'Other': '#9C27B0'},
                    labels={'mean': 'Mean 7-Day Return (%)', 'app_type_clean': 'Application Type'},
                )
                fig_bar.update_layout(height=340, margin=dict(t=20, b=20), showlegend=False)
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### RSS Score vs 7-Day Return")
        if 'rss_score' in df.columns and 'return_7d' in df.columns:
            fig_scatter = px.scatter(
                df,
                x='rss_score',
                y='return_7d',
                color='price_up_7d',
                color_discrete_map={0: '#ef5350', 1: '#66bb6a'},
                hover_data=['ticker', 'app_type_clean'] if 'ticker' in df.columns else None,
                labels={'rss_score': 'RSS Score', 'return_7d': '7-Day % Return',
                        'price_up_7d': 'Stock UP'},
                trendline=None,
                opacity=0.65,
                height=400
            )
            fig_scatter.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

        st.markdown("### Approvals Per Year")
        if 'approval_year' in df.columns and 'app_type_clean' in df.columns:
            yearly = df.groupby(['approval_year', 'app_type_clean']).size().reset_index(name='count')
            fig_yr = px.bar(
                yearly,
                x='approval_year',
                y='count',
                color='app_type_clean',
                color_discrete_map={'NDA': '#2196F3', 'BLA': '#4CAF50', 'ANDA': '#FF9800', 'Other': '#9C27B0'},
                labels={'count': 'Approvals', 'approval_year': 'Year', 'app_type_clean': 'Type'},
                height=380
            )
            fig_yr.update_layout(margin=dict(t=20, b=20))
            st.plotly_chart(fig_yr, use_container_width=True)

    # Show saved result images
    result_images = {
        '01_model_comparison.png': 'Model Comparison',
        '02_roc_curves.png': 'ROC Curves',
        '03_confusion_matrix.png': 'Confusion Matrix',
        '05_rss_vs_movement.png': 'RSS vs Stock Movement',
        '06_finbert_vs_vader.png': 'FinBERT vs VADER',
    }

    st.markdown("---")
    st.markdown("### Saved Result Charts")
    for fname, title in result_images.items():
        fpath = _path("outputs", "results", fname)
        if os.path.exists(fpath):
            st.image(fpath, caption=title, use_column_width=True)


# ============================================================
# TAB: ABOUT
# ============================================================
elif "ABOUT" in tab_selection:
    st.markdown('<div class="main-header">👤 About This Research</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        I'm Hari — a Data Scientist and ML Engineer who gets genuinely curious about problems
        that sit at the edge of two fields nobody thought to combine.

        This project started from a simple question I couldn't stop thinking about: every time
        the FDA approves a drug, pharmaceutical stocks move. But why do some approvals move the
        stock 10% and others barely register? The news is the same format every time. The
        difference is in the regulatory details — what kind of approval, what kind of drug,
        when it happened.

        Most people building stock prediction models are optimizing RSI and MACD. I wanted to
        know if the regulatory structure of an FDA decision — the actual classification, not the
        words — carries a signal that technical indicators completely miss.

        So I built the Regulatory Sentiment Score to find out. The honest answer from the data:
        yes, it adds signal. Not enough to retire on, but enough to be interesting — and enough
        to show that structured regulatory metadata and NLP sentiment are measuring genuinely
        different things.

        I'm targeting Data Scientist and ML Engineer roles in healthcare and finance. This is
        the kind of problem I want to keep working on.

        *MS Artificial Intelligence — Southwest Baptist University, United States*
        """)

    with col2:
        st.markdown("""
        ### Contact

        📧 [hpvykuntapu@gmail.com](mailto:hpvykuntapu@gmail.com)

        💼 [LinkedIn](https://www.linkedin.com/in/harivykuntapu)

        💻 [GitHub](https://github.com/HariVykuntapu)

        📦 [Project Repo](https://github.com/HariVykuntapu/fda-stock-prediction)

        ---

        ### Technical Stack

        - **Data:** OpenFDA API, yfinance
        - **NLP:** FinBERT (ProsusAI), VADER
        - **ML:** scikit-learn, XGBoost
        - **Explainability:** SHAP
        - **Viz:** matplotlib, seaborn, Plotly
        - **App:** Streamlit

        ---

        ### Target Roles

        - Data Scientist (Healthcare / Finance)
        - ML Engineer (NLP / Tabular)

        ---

        ### Academic Context

        MS Artificial Intelligence
        Southwest Baptist University
        United States
        """)

    st.markdown("---")
    st.markdown("""
    ### What Doesn't Work Yet

    **296 samples is the hard ceiling.** I restricted the universe to 12 large-cap names because
    sponsor-to-ticker matching gets messy with smaller companies. That cap limits what any
    classifier can learn. Small-cap biotechs — where a single approval can move the stock 50–200%
    — would show stronger signal, but you'd need a very different data pipeline to get there.

    **The 7-day window is noisy.** Earnings calls, analyst upgrades, macro events — all of it
    contaminates the 7-day return. A 2-day window would give cleaner signal; I chose 7 because
    many approvals drop after market close and need trading days to be fully absorbed.

    **The RSS weights are still a prior.** The 0.45/0.35/0.20 allocation came from reasoning
    about how algorithmic traders vs. fundamental analysts process approval news — not from the
    data. A Bayesian search over the weight space is the obvious next step.

    **What I'm building next:**
    - Run FinBERT on actual 8-K filings instead of constructed text strings
    - Expand to 50+ companies, including mid-cap biotechs where the signal should be stronger
    - Build a portfolio backtest to test whether the UP/DOWN predictions translate to real edge
    """)

    st.markdown("---")
    st.markdown('<div class="footer-text">© 2026 Hari Vykuntapu | MS Artificial Intelligence | Southwest Baptist University, United States</div>', unsafe_allow_html=True)
