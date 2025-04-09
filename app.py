import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier
import calendar

# ----------------------------------------
# 1. Load the trained model and raw data
# ----------------------------------------
st.set_page_config(layout="wide", page_title="Natural Gas Arbitrage Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("clean_features.csv", parse_dates=["Date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()
df = load_data()

features = ["Price", "Spot_Return_1d", "Spot_Vol_5d", "Month", "Lagged_Basis_5d"]
X = df[features]
df["Predicted"] = model.predict(X)
df["Correct"] = (df["Predicted"] == df["Arb_Opportunity"]).astype(int)
df["Year"] = df["Date"].dt.year

# Map month numbers to names
month_name_map = {i: calendar.month_name[i] for i in range(1, 13)}
df["Month_Name"] = df["Month"].map(month_name_map)

# ----------------------------------------
# Sidebar Navigation
# ----------------------------------------
section = st.sidebar.radio("Navigation", ["📊 Dashboard", "📘 Project Overview", "🧪 Methodology"])

# ----------------------------------------
# Project Overview
# ----------------------------------------
if section == "📘 Project Overview":
    st.title("Natural Gas Arbitrage Prediction")
    st.markdown(
        """
        ### Project Overview
        This dashboard analyzes arbitrage opportunities in the U.S. natural gas market by comparing **spot prices** (e.g., Henry Hub) and **futures prices** (e.g., NYMEX contracts).

        A machine learning model was trained to predict when the **spread (basis)** between spot and futures suggests a profitable arbitrage trade — defined here as a **net opportunity > $0.10 per MMBtu**.

        - Green dots indicate correct model predictions (arbitrage was profitable)
        - Red dots indicate false positives (no actual profit)
        - The gray line shows the realized profit per trade (Net_Arb_Profit)
        """
    )

# ----------------------------------------
# Methodology
# ----------------------------------------
elif section == "Methodology":
    st.title("Modeling Methodology")
    st.markdown(
        """
        ### Feature Engineering
        - `Lagged_Basis_5d`: 5-day rolling average of basis (futures - spot)
        - `Spot_Return_1d`, `Fut_Return_1d`: daily returns
        - `Spot_Vol_5d`: 5-day rolling standard deviation of spot price
        - `Month`: seasonal component

        ### Modeling
        - Model: Random Forest Classifier (`sklearn`)
        - Target: 1 if Net Arbitrage Profit > $0.10, else 0
        - Evaluation: Precision, recall, F1-score on test set

        ### Data Sources
        - EIA API: Spot prices (Henry Hub)
        - Yahoo Finance: Futures prices (NG=F)

        All feature engineering and modeling done in Python with `pandas`, `scikit-learn`, and `matplotlib`.
        """
    )

# ----------------------------------------
# Dashboard View
# ----------------------------------------
else:
    st.title("Natural Gas Arbitrage Signal Dashboard")

    st.markdown(
        """
        ### Project Overview
        This dashboard analyzes arbitrage opportunities in the U.S. natural gas market by comparing **spot prices** (e.g., Henry Hub) and **futures prices** (e.g., NYMEX contracts).

        A machine learning model was trained to predict when the **spread (basis)** between spot and futures suggests a profitable arbitrage trade — defined here as a **net opportunity > $0.10 per MMBtu**.

        - Green dots indicate correct model predictions (arbitrage was profitable)
        - Red dots indicate false positives (no actual profit)
        - The gray line shows the realized profit per trade (Net_Arb_Profit)
        ---
        """
    )

    st.subheader("Set Threshold and Filters")

    arb_threshold = st.slider(
        "Set Arbitrage Threshold ($/MMBtu)", min_value=0.0, max_value=1.0, value=0.10, step=0.01
    )

    month_options = sorted(df["Month_Name"].unique(), key=lambda x: list(calendar.month_name).index(x))
    selected_months = st.multiselect("Filter by Month", month_options, default=month_options)

    year_range = st.slider(
        "Select Year Range",
        min_value=int(df["Year"].min()),
        max_value=int(df["Year"].max()),
        value=(int(df["Year"].min()), int(df["Year"].max())),
        step=1
    )

    prediction_filter = st.selectbox("Filter by Model Signal", ["All", "Predicted Arbitrage Only", "No Arbitrage Only"])

    show_signals = st.checkbox("Show Model Prediction Markers", value=True)

    # Apply filters
    filtered_df = df[
        df["Month_Name"].isin(selected_months) &
        df["Year"].between(year_range[0], year_range[1])
    ]

    if prediction_filter == "Predicted Arbitrage Only":
        filtered_df = filtered_df[filtered_df["Predicted"] == 1]
    elif prediction_filter == "No Arbitrage Only":
        filtered_df = filtered_df[filtered_df["Predicted"] == 0]

    st.subheader("Net Arbitrage Profit with Model Predictions")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(filtered_df["Date"], filtered_df["Net_Arb_Profit"], label="Net Arbitrage Profit", color="lightgray", linewidth=1.5)

    if show_signals:
        correct_preds = filtered_df[filtered_df["Predicted"] == 1]
        colors = correct_preds["Correct"].map({1: "green", 0: "red"})
        ax.scatter(
            correct_preds["Date"],
            correct_preds["Net_Arb_Profit"],
            c=colors,
            s=40,
            label="Model Signal (Green = Correct, Red = False Alarm)",
            alpha=0.8,
        )

    ax.axhline(arb_threshold, linestyle="--", color="gray", linewidth=1, label=f"Arb Threshold (${arb_threshold:.2f})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net Arbitrage Profit ($/MMBtu)")
    ax.set_title("Arbitrage Predictions Over Time", pad=20)
    ax.set_ylim(-1, 2)  # Optional: tighten y-axis
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Model Feature Importances")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
    feat_df = feat_df.sort_values("Importance", ascending=False)
    st.dataframe(feat_df.reset_index(drop=True))
