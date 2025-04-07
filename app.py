import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestClassifier

# 1. Load the trained model and raw data
st.set_page_config(layout="wide")
st.title("Natural Gas Arbitrage Signal Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("clean_features.csv", parse_dates=["Date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

model = load_model()
df = load_data()

# 2. Predictions
features = ["Price", "Spot_Return_1d", "Spot_Vol_5d", "Month", "Lagged_Basis_5d"]
X = df[features]
df["Predicted"] = model.predict(X)
df["Correct"] = (df["Predicted"] == df["Arb_Opportunity"]).astype(int)



#  3. Plot Net Arbitrage Profit with Signals

#DROPDOWN FILTER###
month_options = sorted(df["Month"].unique())
selected_months = st.multiselect("Filter by Month", month_options, default=month_options)

# Dropdown: Filter by year
df["Year"] = df["Date"].dt.year

min_year = df["Year"].min()
max_year = df["Year"].max()

year_range = st.slider(
    "Select Year Range",
    min_value=int(min_year),
    max_value=int(max_year),
    value=(int(min_year), int(max_year)),
    step=1
)

# Dropdown: Filter by model prediction
prediction_filter = st.selectbox("Filter by Model Signal", ["All", "Predicted Arbitrage Only", "No Arbitrage Only"])

# Apply filters
filtered_df = df[
    df["Month"].isin(selected_months) &
    df["Year"].between(year_range[0], year_range[1])
]
if prediction_filter == "Predicted Arbitrage Only":
    filtered_df = filtered_df[filtered_df["Predicted"] == 1]
elif prediction_filter == "No Arbitrage Only":
    filtered_df = filtered_df[filtered_df["Predicted"] == 0]
######

st.subheader("Net Arbitrage Profit with Model Predictions")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(filtered_df["Date"], filtered_df["Net_Arb_Profit"], label="Net Arbitrage Profit", color="lightgray")

correct_preds = filtered_df[filtered_df["Predicted"] == 1]
colors = correct_preds["Correct"].map({1: "green", 0: "red"})

ax.scatter(
    correct_preds["Date"],
    correct_preds["Net_Arb_Profit"],
    c=colors,
    label="Model Signal (Green = Correct, Red = False Alarm)",
    alpha=0.8,
)

ax.axhline(0.1, linestyle="--", color="gray", linewidth=1, label="Arb Threshold ($0.10)")
ax.set_xlabel("Date")
ax.set_ylabel("Net Arbitrage Profit ($/MMBtu)")
ax.set_title("Arbitrage Predictions Over Time")
ax.legend()
ax.grid(True)
q1 = filtered_df["Net_Arb_Profit"].quantile(0.01)
q99 = filtered_df["Net_Arb_Profit"].quantile(0.99)
ax.set_ylim(q1, q99)
st.pyplot(fig)

# 4. Optional: Show feature importances
# 
st.subheader("Model Feature Importances")
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_df = feat_df.sort_values("Importance", ascending=False)
st.dataframe(feat_df.reset_index(drop=True))
