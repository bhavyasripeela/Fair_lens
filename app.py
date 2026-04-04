import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FairLens AI",
    page_icon="🧠",
    layout="wide"
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
    <style>
        .main-title {
            font-size: 2.8rem;
            font-weight: 800;
            color: inherit;
        }
        .subtitle {
            font-size: 1.1rem;
            color: inherit;
            opacity: 0.7;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: rgba(100, 149, 237, 0.1);
            border-radius: 12px;
            padding: 1rem 1.5rem;
            text-align: center;
        }
        .section-header {
            font-size: 1.3rem;
            font-weight: 700;
            color: inherit;
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
        .stAlert > div {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🧠 FairLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Fairness Auditor — Detect · Explain · Fix Bias in AI Systems</div>', unsafe_allow_html=True)
st.divider()

# ─── Domain Selection ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 1 — Select Application Domain</div>', unsafe_allow_html=True)

DOMAIN_CONFIG = {
    "Hiring": {
        "hint": "Expected columns: `gender`, `experience`, `score`, `selected`",
        "group_col": "gender",
        "outcome_col": "selected",
        "outcome_label": "Selection Rate",
    },
    "Loan": {
        "hint": "Expected columns: `gender`, `income`, `credit_score`, `loan_status`",
        "group_col": "gender",
        "outcome_col": "loan_status",
        "outcome_label": "Approval Rate",
    },
    "Education": {
        "hint": "Expected columns: `gender`, `marks`, `category`, `admitted`",
        "group_col": "gender",
        "outcome_col": "admitted",
        "outcome_label": "Admission Rate",
    },
    "Custom": {
        "hint": "Upload any dataset with at least one categorical column for bias detection.",
        "group_col": None,
        "outcome_col": None,
        "outcome_label": "Outcome Rate",
    },
}

col_domain, col_hint = st.columns([1, 2])

with col_domain:
    domain = st.selectbox(
        "Choose Application Domain",
        list(DOMAIN_CONFIG.keys()),
        label_visibility="collapsed"
    )

with col_hint:
    st.info(f"📌 {DOMAIN_CONFIG[domain]['hint']}")

st.divider()

# ─── File Upload ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 2 — Upload Your Dataset</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"],
    label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("⬆️ Please upload a CSV file to begin the fairness analysis.")
    st.stop()

# ─── Load Data ────────────────────────────────────────────────────────────────
try:
    df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"❌ Error reading file: {e}")
    st.stop()

st.success(f"✅ File uploaded successfully — {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("📋 Preview Dataset", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()

# ─── Column Selection ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 3 — Select Column for Bias Detection</div>', unsafe_allow_html=True)

categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
all_cols = df.columns.tolist()

if not categorical_cols:
    st.warning("⚠️ No categorical columns found. Showing all columns.")
    available_cols = all_cols
else:
    available_cols = categorical_cols

col_a, col_b = st.columns(2)

with col_a:
    default_group = DOMAIN_CONFIG[domain]["group_col"]
    group_default_idx = available_cols.index(default_group) if default_group in available_cols else 0
    group_col = st.selectbox("Sensitive Attribute (e.g. gender, race)", available_cols, index=group_default_idx)

with col_b:
    default_outcome = DOMAIN_CONFIG[domain]["outcome_col"]
    numeric_cols = df.select_dtypes(include=["number", "bool", "object"]).columns.tolist()
    outcome_default_idx = numeric_cols.index(default_outcome) if default_outcome in numeric_cols else 0
    outcome_col = st.selectbox("Outcome Column (e.g. selected, approved)", numeric_cols, index=outcome_default_idx)

st.divider()

# ─── Distribution Analysis ────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 4 — Distribution Analysis</div>', unsafe_allow_html=True)

value_counts = df[group_col].value_counts()
total = value_counts.sum()
percentages = (value_counts / total * 100).round(2)

col_chart, col_stats = st.columns([2, 1])

with col_chart:
    fig_dist = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        labels={"x": group_col.title(), "y": "Count"},
        title=f"Distribution of '{group_col}'",
        color=value_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set2,
        text=value_counts.values,
    )
    fig_dist.update_traces(textposition="outside")
    fig_dist.update_layout(
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(t=50, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_dist, use_container_width=True)

with col_stats:
    st.markdown("**Group Breakdown**")
    for group, pct in percentages.items():
        st.metric(label=str(group), value=f"{pct:.1f}%", delta=f"{value_counts[group]} records")

    if len(percentages) == 2:
        diff = abs(percentages.iloc[0] - percentages.iloc[1])
        rep_score = round(100 - diff, 2)
        st.markdown("---")
        st.metric("Representation Balance", f"{rep_score:.1f}%", delta="Fair" if diff <= 20 else "Imbalanced")

st.divider()

# ─── Fairness Analysis ────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Step 5 — Fairness Analysis</div>', unsafe_allow_html=True)

outcome_label = DOMAIN_CONFIG[domain]["outcome_label"]

# Detect binary outcome column
def to_binary(series):
    """Convert yes/no, 1/0, true/false column to numeric 0-1."""
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            return series.astype(float)
        return None
    str_series = series.str.lower().str.strip()
    yes_vals = {"yes", "1", "true", "approved", "selected", "admitted", "accept", "pass"}
    if str_series.isin(yes_vals | {"no", "0", "false", "rejected", "not selected", "denied", "fail"}).all():
        return str_series.isin(yes_vals).astype(float)
    return None

binary_outcome = to_binary(df[outcome_col]) if outcome_col in df.columns else None

if binary_outcome is not None and group_col in df.columns:
    analysis_df = df[[group_col]].copy()
    analysis_df["__outcome__"] = binary_outcome.values

    outcome_rates = (
        analysis_df.groupby(group_col)["__outcome__"]
        .mean()
        .mul(100)
        .round(2)
        .sort_values(ascending=False)
    )

    col_f1, col_f2 = st.columns([2, 1])

    with col_f1:
        fig_rates = px.bar(
            x=outcome_rates.index,
            y=outcome_rates.values,
            labels={"x": group_col.title(), "y": f"{outcome_label} (%)"},
            title=f"{outcome_label} by {group_col.title()}",
            color=outcome_rates.index,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            text=[f"{v:.1f}%" for v in outcome_rates.values],
        )
        fig_rates.update_traces(textposition="outside")
        fig_rates.update_layout(
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(range=[0, min(outcome_rates.max() * 1.3, 100)]),
            margin=dict(t=50, b=20, l=20, r=20),
        )
        st.plotly_chart(fig_rates, use_container_width=True)

    with col_f2:
        max_rate = outcome_rates.max()
        min_rate = outcome_rates.min()
        dir_score = round(min_rate / max_rate, 4) if max_rate > 0 else 1.0
        parity_diff = round(max_rate - min_rate, 2)

        st.markdown("**Fairness Metrics**")
        st.metric("Disparate Impact Ratio", f"{dir_score:.2f}", delta="≥ 0.80 is fair")
        st.metric("Statistical Parity Gap", f"{parity_diff:.1f}%", delta="≤ 10% is fair")
        st.metric("Highest Rate", f"{max_rate:.1f}%")
        st.metric("Lowest Rate", f"{min_rate:.1f}%")

    # ─── Verdict ──────────────────────────────────────────────────────────────
    st.markdown("#### 🧾 Fairness Verdict")

    passes_dir = dir_score >= 0.80
    passes_parity = parity_diff <= 10

    v1, v2 = st.columns(2)
    with v1:
        if passes_dir:
            st.success(f"✅ Disparate Impact Ratio = {dir_score:.2f} — Passes the 80% rule.")
        else:
            st.error(f"⚠️ Disparate Impact Ratio = {dir_score:.2f} — Fails the 80% rule. Potential discrimination risk.")

    with v2:
        if passes_parity:
            st.success(f"✅ Statistical Parity Gap = {parity_diff:.1f}% — Within acceptable range.")
        else:
            st.error(f"⚠️ Statistical Parity Gap = {parity_diff:.1f}% — Significant disparity detected across groups.")

    if passes_dir and passes_parity:
        st.balloons()
        st.success("🎉 Overall: No major bias detected. The model appears to be fair across groups.")
    else:
        st.error("🚨 Overall: Bias detected. Review your training data and model pipeline before deployment.")

else:
    st.warning(
        f"⚠️ Could not compute outcome-based fairness for column `{outcome_col}`. "
        "Make sure it contains binary values (yes/no, 1/0, true/false, approved/rejected, etc.). "
        "Showing representation-only analysis instead."
    )

    if len(percentages) == 2:
        diff = abs(percentages.iloc[0] - percentages.iloc[1])
        fairness_score = round(100 - diff, 2)
        st.metric("Representation Fairness Score", f"{fairness_score:.1f}%")

        if diff > 20:
            st.error("⚠️ Significant group imbalance detected in the dataset.")
        else:
            st.success("✅ Groups are roughly balanced. No major representation bias.")
    else:
        st.info("ℹ️ Fairness scoring works best with 2 groups. Multi-group support coming soon.")

st.divider()

# ─── Footer ───────────────────────────────────────────────────────────────────
st.caption("FairLens AI · Built for AI/ML Hackathon 2025 · Bhavya Sri · Ruchitha · Tulasi Priya · Vaishnavi")


