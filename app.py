import streamlit as st
import pandas as pd
import plotly.express as px

# Title
st.title("FairLens AI 🧠")
st.subheader("AI Fairness Auditor")
st.subheader("Select Domain")

domain = st.selectbox(
    "Choose Application Domain",
    ["Hiring", "Loan", "Education", "Custom"]
)
if domain == "Hiring":
    st.info("📌 Expected columns: gender, experience, score, selected")

elif domain == "Loan":
    st.info("📌 Expected columns: gender, income, credit_score, loan_status")

elif domain == "Education":
    st.info("📌 Expected columns: gender, marks, category, admitted")

else:
    st.info("📌 Upload any dataset with at least one categorical column for bias detection")

# Upload file
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

# Check if file is uploaded
if uploaded_file is not None:
    try:
        # Read dataset
        df = pd.read_csv(uploaded_file)

        st.success("File uploaded successfully!")

        # Show preview
        st.write("### Dataset Preview")
        st.dataframe(df.head())

        # -------------------------------
        # STEP 3 — Column Selection
        # -------------------------------
        st.subheader("Select Column for Bias Detection")

# Select only categorical columns

        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

        if len(categorical_cols) == 0:
            st.warning("⚠️ No categorical columns found. Showing all columns instead.")
            column = st.selectbox("Choose a column", df.columns)
        else:
            column = st.selectbox("Choose a column", categorical_cols)

        if column:
            st.write(f"### Distribution of {column}")

            # Count values
            value_counts = df[column].value_counts()

            # Show counts
            st.write(value_counts)

            # Plot graph
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                labels={'x': column, 'y': 'Count'},
                title=f"{column} Distribution"
            )

            st.plotly_chart(fig)

            # -------------------------------
            # STEP 4 — Fairness Score
            # -------------------------------
            st.subheader("Fairness Analysis")

            total = value_counts.sum()
            percentages = (value_counts / total) * 100

            st.write("### Percentage Distribution")
            st.write(percentages)

            # Fairness score only if 2 groups
            if len(percentages) == 2:
                diff = abs(percentages.iloc[0] - percentages.iloc[1])
                fairness_score = 100 - diff

                st.write(f"### Fairness Score: {fairness_score:.2f}%")

                # Bias alert
                if diff > 20:
                    st.error("⚠️ Bias Detected! Significant imbalance found.")
                else:
                    st.success("✅ Fair distribution. No major bias detected.")

            else:
                st.info("Fairness score works best for 2 groups ")

    except Exception as e:
        st.error(f"Error reading file: {e}")