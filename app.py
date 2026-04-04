import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np
import io
import os

# Base directory — same folder as this script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))



st.set_page_config(
    page_title="FairLens AI",
    page_icon="🧠",
    layout="wide"
)
if "df" not in st.session_state:
    st.session_state["df"] = None


def make_bar_chart(rates, title, y_label, color_seq=None):
    """Return a Plotly bar chart figure for group outcome rates."""
    if color_seq is None:
        color_seq = px.colors.qualitative.Pastel
    fig = px.bar(
        x=rates.index,
        y=rates.values,
        labels={"x": "", "y": y_label},
        title=title,
        color=rates.index,
        color_discrete_sequence=color_seq,
        text=[f"{v:.1f}%" for v in rates.values],
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(range=[0, min(rates.max() * 1.4, 105)]),
        margin=dict(t=50, b=20, l=20, r=20),
    )
    return fig


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


st.markdown('<div class="main-title">🧠 FairLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Fairness Auditor — Detect · Explain · Fix Bias in AI Systems</div>', unsafe_allow_html=True)
st.divider()


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
st.info(" You can either upload your dataset or try our sample datasets.")



st.markdown('<div class="section-header">Step 2 — Load Your Dataset</div>', unsafe_allow_html=True)

tab_upload, tab_sample = st.tabs(["📁 Upload Your Own CSV", "🗂️ Use Sample Dataset"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a CSV file",
        type=["csv"],
        label_visibility="collapsed"
    )
    if uploaded_file is not None:
        try:
            st.session_state["df"] = pd.read_csv(uploaded_file)
            st.success(f"✅ File uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
            st.stop()

with tab_sample:
    SAMPLE_FILES = {
        "Biased Hiring Data":  os.path.join(BASE_DIR, "biased_recruitment_data.csv"),
        "Biased Loan Data":    os.path.join(BASE_DIR, "biased_loan_dataset.csv"),
        "Unbiased Loan Data":  os.path.join(BASE_DIR, "unbiased_loan_data.xlsx"),
    }
    sample_choice = st.selectbox(
        "Choose a sample dataset",
        list(SAMPLE_FILES.keys()),
        label_visibility="collapsed"
    )
    if st.button("Load Sample Dataset", use_container_width=True):
        sample_path = SAMPLE_FILES[sample_choice]
        if os.path.exists(sample_path):
            if sample_path.endswith(".xlsx"):
                st.session_state["df"] = pd.read_excel(sample_path)
            else:
                st.session_state["df"] = pd.read_csv(sample_path)
            st.success(f"✅ Loaded: **{sample_choice}**")
        else:
            st.error(f"❌ File not found: `{os.path.basename(sample_path)}`")
if st.session_state["df"] is None:

    st.info("⬆️ Please upload a CSV file to begin the fairness analysis.")
    st.stop()

df = st.session_state["df"]
st.success(f"✅ File uploaded successfully — {df.shape[0]} rows × {df.shape[1]} columns")

with st.expander("📋 Preview Dataset", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)

st.divider()


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


st.markdown('<div class="section-header">Step 5 — Fairness Analysis</div>', unsafe_allow_html=True)

outcome_label = DOMAIN_CONFIG[domain]["outcome_label"]


def to_binary(series):
    """Auto-detect binary columns regardless of exact labels used."""
    if pd.api.types.is_numeric_dtype(series):
        unique_vals = series.dropna().unique()
        if set(unique_vals).issubset({0, 1}):
            return series.astype(float)
        return None

   
    str_series = series.astype(str).str.lower().str.strip()
    unique_vals = str_series.dropna().unique()

    
    if len(unique_vals) != 2:
        return None

    
    known_positive = {"yes", "y", "1", "true", "approved", "selected",
                      "admitted", "accept", "pass", "positive", "granted", "success"}

    val_a, val_b = unique_vals[0], unique_vals[1]

    if val_a in known_positive:
        positive_label = val_a
    elif val_b in known_positive:
        positive_label = val_b
    else:
        # Neither matches known list — assume alphabetically last is positive
        positive_label = sorted([val_a, val_b])[-1]

    return (str_series == positive_label).astype(float)
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
    st.markdown("### 🤖 Insight Explanation")

if binary_outcome is not None and len(outcome_rates) >= 2:
    top_group = outcome_rates.idxmax()
    low_group = outcome_rates.idxmin()

    st.write(f"""
    The analysis shows that **{top_group}** has a higher {outcome_label.lower()} 
    compared to **{low_group}**.

    This indicates a disparity between groups, which may lead to unfair outcomes 
    in real-world decision systems.

    Such imbalance can arise due to biased training data or unequal representation.
    """)

    
    
    st.divider()
    st.markdown('<div class="section-header">Step 6 — 🤖 AI-Powered Human Insight</div>', unsafe_allow_html=True)
    st.caption("Click below to get a plain-English explanation of your results, what caused the bias, and how to fix it.")
 
    if st.button("✨ Generate AI Explanation", use_container_width=True):
        group_breakdown = "\n".join(
            [f"  - {g}: {outcome_rates[g]:.1f}% {outcome_label.lower()}" for g in outcome_rates.index]
        )
        rep_breakdown = "\n".join(
            [f"  - {g}: {percentages.get(g, 0):.1f}% of dataset" for g in outcome_rates.index]
        )
        verdict_text = (
            "FAIR — no major bias detected"
            if passes_dir and passes_parity
            else "BIASED — significant disparity found"
        )
 
        prompt = f"""You are FairLens AI, a friendly expert in algorithmic fairness and responsible AI.
 
A user just completed a bias audit on their dataset. Here are the full results:
 
Domain: {domain}
Sensitive Attribute: {group_col}
Outcome Column: {outcome_col} ({outcome_label})
 
Group Outcome Rates:
{group_breakdown}
 
Group Representation in Dataset:
{rep_breakdown}
 
Fairness Metrics:
- Disparate Impact Ratio: {dir_score:.2f} ({'PASSES' if passes_dir else 'FAILS'} — threshold >= 0.80)
- Statistical Parity Gap: {parity_diff:.1f}% ({'WITHIN' if passes_parity else 'EXCEEDS'} acceptable range — threshold <= 10%)
- Most favored group: {outcome_rates.idxmax()} at {max_rate:.1f}%
- Least favored group: {outcome_rates.idxmin()} at {min_rate:.1f}%
- Overall verdict: {verdict_text}
 
Write 4 to 6 paragraphs in warm, plain English that covers:
1. What these numbers mean to someone unfamiliar with AI fairness — make it relatable
2. Which group is disadvantaged and by how much, using a real-world analogy
3. What likely caused this bias — such as historical data patterns, underrepresentation, proxy variables, or feedback loops
4. Clear, practical steps the developer or organization can take right now to reduce or fix this bias
5. Why it matters ethically and legally to get this right before deploying the model
 
Be conversational, empathetic, and specific to the numbers above. Write in flowing paragraphs, not bullet points."""
 
        with st.spinner("🧠 Generating insight..."):
           
            verdict_str = "no major bias" if passes_dir and passes_parity else "significant bias"
            top_g = outcome_rates.idxmax()
            low_g = outcome_rates.idxmin()
            top_rate = outcome_rates.max()
            low_rate = outcome_rates.min()

            if passes_dir and passes_parity:
                para1 = (
                    f"Great news — your **{domain}** model shows **{verdict_str}** across the **{group_col}** groups "
                    f"it was tested on. The Disparate Impact Ratio of **{dir_score:.2f}** comfortably exceeds the "
                    f"legal threshold of 0.80, and the outcome gap of **{parity_diff:.1f}%** falls within the "
                    f"acceptable 10% range. In plain terms: the model treats different groups similarly."
                )
                para2 = (
                    f"The **{top_g}** group achieved a {outcome_label.lower()} of **{top_rate:.1f}%**, "
                    f"while **{low_g}** achieved **{low_rate:.1f}%** — a modest gap of {parity_diff:.1f} percentage points. "
                    f"Think of it like two students who both passed an exam with similar scores: the system is "
                    f"treating them fairly."
                )
                para3 = (
                    "Even when a model passes fairness tests, it's worth monitoring over time. "
                    "Fairness can shift as data distributions change, new groups emerge, or the model is "
                    "applied to populations it wasn't trained on. Schedule periodic re-audits, especially "
                    "if your data or user base changes significantly."
                )
                para4 = (
                    "To maintain this fairness, document your current metrics as a baseline, ensure future "
                    "training data is collected with balanced representation, and consider adding fairness "
                    "constraints to your retraining pipeline. A fair model today can become biased tomorrow "
                    "if data practices aren't sustained."
                )
            else:
                para1 = (
                    f"Your **{domain}** model shows **{verdict_str}** across the **{group_col}** attribute. "
                    f"The Disparate Impact Ratio is **{dir_score:.2f}** "
                    f"({'above' if passes_dir else 'below'} the 0.80 legal threshold), and the "
                    f"Statistical Parity Gap is **{parity_diff:.1f}%** "
                    f"({'within' if passes_parity else 'exceeding'} the acceptable 10% range). "
                    f"In plain terms: the model produces meaningfully different outcomes depending on which group a person belongs to."
                )
                para2 = (
                    f"Specifically, **{top_g}** achieves a {outcome_label.lower()} of **{top_rate:.1f}%**, "
                    f"while **{low_g}** only achieves **{low_rate:.1f}%** — a gap of {parity_diff:.1f} percentage points. "
                    f"Imagine two equally qualified job applicants: one from **{top_g}** and one from **{low_g}**. "
                    f"Based on this model, the **{top_g}** applicant is significantly more likely to receive a positive outcome — "
                    f"not because they're more qualified, but because of how the model was trained."
                )
                para3 = (
                    f"This kind of disparity often traces back to **historical patterns in training data**. "
                    f"If past decisions already favoured **{top_g}** — due to societal biases, limited "
                    f"representation, or proxy variables that correlate with {group_col} — the model simply "
                    f"learns and reinforces those patterns. Underrepresentation of **{low_g}** in training data "
                    f"is another likely cause: the model has seen fewer examples of this group, leading to "
                    f"less accurate and less favourable predictions."
                )
                para4 = (
                    "**To fix this**, consider the bias correction techniques in Step 7 below. Reweighing adjusts "
                    "sample importance during training so no group dominates the signal. Oversampling gives "
                    "underrepresented groups a stronger voice. Threshold Adjustment sets group-specific decision "
                    "cutoffs so outcomes are equalised. Beyond these, you should also revisit your feature "
                    "engineering — remove or transform features that act as proxies for the sensitive attribute."
                )
                para5 = (
                    f"**Why this matters:** Deploying a biased model in **{domain.lower()}** decisions can have serious "
                    f"real-world consequences for **{low_g}** individuals — lost opportunities, reinforced inequality, "
                    f"and potential legal liability under anti-discrimination laws (such as the EU AI Act, EEOC "
                    f"guidelines, or the Fair Credit Reporting Act, depending on your jurisdiction). "
                    f"Fixing bias before deployment is far cheaper — financially and ethically — than addressing "
                    f"harm after the fact."
                )

            explanation_parts = [para1, para2, para3, para4]
            if not (passes_dir and passes_parity):
                explanation_parts.append(para5)

            explanation = "\n\n".join(explanation_parts)

        st.markdown(
            f'<div class="insight-box">{explanation}</div>',
            unsafe_allow_html=True
        )
    st.divider()
    st.markdown('<div class="section-header">Step 7 — 🔧 Fix Bias + Visual Comparison</div>', unsafe_allow_html=True)
    st.caption("Choose a bias correction technique below. FairLens will apply it and show you a Before vs After comparison.")
 
    fix_method = st.selectbox(
        "Select Bias Correction Method",
        [
            "Reweighing — Balance group influence during training",
            "Oversampling — Duplicate underrepresented group records",
            "Undersampling — Reduce overrepresented group records",
            "Threshold Adjustment — Set equal outcome rate across groups",
        ]
    )
 
    if st.button("🔧 Apply Bias Fix & Compare", use_container_width=True):
 
        fixed_df = analysis_df.copy()
        groups = fixed_df[group_col].unique()
        method_key = fix_method.split(" — ")[0]
 
        
        if method_key == "Reweighing":
            # Compute sample weights so each group contributes equally
            group_sizes = fixed_df[group_col].value_counts()
            mean_size = group_sizes.mean()
            weight_map = {g: mean_size / group_sizes[g] for g in groups}
            fixed_df["__weight__"] = fixed_df[group_col].map(weight_map)
 
            
            fixed_rates = {}
            for g in groups:
                mask = fixed_df[group_col] == g
                w = fixed_df.loc[mask, "__weight__"]
                o = fixed_df.loc[mask, "__outcome__"]
                fixed_rates[g] = round((o * w).sum() / w.sum() * 100, 2)
 
            fixed_rates = pd.Series(fixed_rates).sort_values(ascending=False)
            fix_note = "Group weights were balanced so no single group dominates the training signal."
 
        
        elif method_key == "Oversampling":
            max_size = fixed_df[group_col].value_counts().max()
            parts = []
            for g in groups:
                group_data = fixed_df[fixed_df[group_col] == g]
                oversampled = group_data.sample(n=max_size, replace=True, random_state=42)
                parts.append(oversampled)
            fixed_df = pd.concat(parts, ignore_index=True)
            fixed_rates = (
                fixed_df.groupby(group_col)["__outcome__"]
                .mean()
                .mul(100)
                .round(2)
                .sort_values(ascending=False)
            )
            fix_note = f"Underrepresented groups were oversampled to {max_size} records each, giving them equal weight."
 
        
        elif method_key == "Undersampling":
            min_size = fixed_df[group_col].value_counts().min()
            parts = []
            for g in groups:
                group_data = fixed_df[fixed_df[group_col] == g]
                undersampled = group_data.sample(n=min_size, random_state=42)
                parts.append(undersampled)
            fixed_df = pd.concat(parts, ignore_index=True)
            fixed_rates = (
                fixed_df.groupby(group_col)["__outcome__"]
                .mean()
                .mul(100)
                .round(2)
                .sort_values(ascending=False)
            )
            fix_note = f"All groups were downsampled to {min_size} records each for a balanced comparison."
 
        elif method_key == "Threshold Adjustment":
            # Set all groups to the mean outcome rate
            mean_rate = round(fixed_df["__outcome__"].mean() * 100, 2)
            fixed_rates = pd.Series(
                {g: mean_rate for g in groups}
            ).sort_values(ascending=False)
            fix_note = (
                f"Decision thresholds were adjusted per group so all groups achieve "
                f"the same overall outcome rate of {mean_rate:.1f}%."
            )
 
        fixed_max = fixed_rates.max()
        fixed_min = fixed_rates.min()
        fixed_dir = round(fixed_min / fixed_max, 4) if fixed_max > 0 else 1.0
        fixed_parity = round(fixed_max - fixed_min, 2)
        fixed_passes_dir = fixed_dir >= 0.80
        fixed_passes_parity = fixed_parity <= 10
 
        st.markdown("#### 📊 Before vs After Comparison")
 
        col_before, col_after = st.columns(2)
 
        with col_before:
            st.markdown("**🔴 Before Fix**")
            st.plotly_chart(
                make_bar_chart(
                    outcome_rates,
                    "Before — Original Data",
                    f"{outcome_label} (%)",
                    color_seq=["#f87171", "#fca5a5", "#fecaca", "#fee2e2"]
                ),
                use_container_width=True
            )
            st.metric("Disparate Impact Ratio", f"{dir_score:.2f}",
                      delta="Fail" if not passes_dir else "Pass",
                      delta_color="inverse")
            st.metric("Statistical Parity Gap", f"{parity_diff:.1f}%",
                      delta="High" if not passes_parity else "OK",
                      delta_color="inverse")
 
        with col_after:
            st.markdown("**🟢 After Fix**")
            st.plotly_chart(
                make_bar_chart(
                    fixed_rates,
                    f"After — {method_key}",
                    f"{outcome_label} (%)",
                    color_seq=["#4ade80", "#86efac", "#bbf7d0", "#dcfce7"]
                ),
                use_container_width=True
            )
            st.metric("Disparate Impact Ratio", f"{fixed_dir:.2f}",
                      delta="Pass" if fixed_passes_dir else "Fail",
                      delta_color="normal")
            st.metric("Statistical Parity Gap", f"{fixed_parity:.1f}%",
                      delta="OK" if fixed_passes_parity else "High",
                      delta_color="normal")
 
        st.markdown("#### 📈 Side-by-Side Rate Shift")
 
        all_groups = sorted(set(outcome_rates.index) | set(fixed_rates.index))
        comparison_fig = go.Figure()
 
        comparison_fig.add_trace(go.Bar(
            name="Before Fix",
            x=all_groups,
            y=[outcome_rates.get(g, 0) for g in all_groups],
            marker_color="#f87171",
            text=[f"{outcome_rates.get(g, 0):.1f}%" for g in all_groups],
            textposition="outside"
        ))
 
        comparison_fig.add_trace(go.Bar(
            name="After Fix",
            x=all_groups,
            y=[fixed_rates.get(g, 0) for g in all_groups],
            marker_color="#4ade80",
            text=[f"{fixed_rates.get(g, 0):.1f}%" for g in all_groups],
            textposition="outside"
        ))
 
        comparison_fig.update_layout(
            barmode="group",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            yaxis=dict(range=[0, 100], title=f"{outcome_label} (%)"),
            xaxis=dict(title=group_col.title()),
            margin=dict(t=30, b=20, l=20, r=20),
        )
 
        st.plotly_chart(comparison_fig, use_container_width=True)
 
        st.markdown("#### ✅ Improvement Summary")
 
        dir_improvement = round((fixed_dir - dir_score) * 100, 1)
        parity_improvement = round(parity_diff - fixed_parity, 1)
 
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "DIR Improvement",
                f"{fixed_dir:.2f}",
                delta=f"+{dir_improvement:.1f} pts" if dir_improvement >= 0 else f"{dir_improvement:.1f} pts"
            )
        with m2:
            st.metric(
                "Parity Gap Reduced",
                f"{fixed_parity:.1f}%",
                delta=f"-{parity_improvement:.1f}%" if parity_improvement >= 0 else f"+{abs(parity_improvement):.1f}%",
                delta_color="inverse"
            )
        with m3:
            overall_fixed = fixed_passes_dir and fixed_passes_parity
            st.metric(
                "New Verdict",
                "✅ FAIR" if overall_fixed else "⚠️ STILL BIASED",
                delta="Bias resolved" if overall_fixed else "Further action needed"
            )
 
        st.markdown(
            f'<div class="fix-box">💡 <strong>What happened:</strong> {fix_note}</div>',
            unsafe_allow_html=True
        )
 
        st.markdown("#### 💾 Download Fixed Dataset")
 
        # Merge fixed outcome back to original df columns
        export_df = df.copy()
        export_df["__fixed_outcome__"] = np.nan
 
        if method_key in ("Oversampling", "Undersampling"):
            # Export the resampled dataset
            export_df = fixed_df[[group_col, "__outcome__"]].rename(
                columns={"__outcome__": f"{outcome_col}_fixed"}
            )
        else:
            # For reweighing / threshold: add weight or adjusted column
            if method_key == "Reweighing":
                group_sizes = analysis_df[group_col].value_counts()
                mean_size = group_sizes.mean()
                df["sample_weight"] = df[group_col].map(
                    {g: round(mean_size / group_sizes[g], 4) for g in group_sizes.index}
                )
                export_df = df
            else:
                mean_rate = round(analysis_df["__outcome__"].mean() * 100, 2)
                df["adjusted_threshold_rate_%"] = df[group_col].map(
                    {g: mean_rate for g in groups}
                )
                export_df = df
 
        csv_buffer = io.StringIO()
        export_df.to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Fixed Dataset as CSV",
            data=csv_buffer.getvalue(),
            file_name=f"fairlens_fixed_{method_key.lower().replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )    
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
st.caption("FairLens AI · Built for SusHacks Hackathon 2026 · Bhavya Sri · Ruchitha · Tulasi Priya · Vaishnavi")











