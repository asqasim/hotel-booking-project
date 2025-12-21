# src/app.py
"""
Streamlit app for Hotel Booking Cancellation Project (IDS f24)

- Uses relative paths (from src/):
    ../data/hotel_bookings.csv
    ../models/hotel_cancellation_model.pkl

Features:
- Home: intro + dataset overview
- EDA: interactive analysis (Lead Time trends, Market Segments, Correlation maps, etc.)
- Predictor (Clean): minimal, clean input with sliders & selects; shows probability (%) and label
- Key Takeaways: concise findings from EDA & modeling
- Robust path resolution and safe handling of mixed-type categorical columns
- Caching: @st.cache_data for data, @st.cache_resource for model
"""

import os
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

warnings.filterwarnings("ignore")

# ------------------------
# Page config & styling
# ------------------------
st.set_page_config(page_title="Hotel Booking Cancellation — IDS f24", page_icon="🏨", layout="wide")

st.markdown(
    """
    <style>
      .title { font-size:34px !important; font-weight:700; color:#0f4c81; }
      .subtitle { font-size:15px !important; color:#333333; margin-bottom:8px; }
      .card { background: linear-gradient(180deg, #fff, #f7fbff); padding: 12px; border-radius: 10px; box-shadow: 0 2px 8px rgba(15,76,129,0.06); }
      .small-muted { font-size:12px; color:#6b7280; }
      .stButton>button { border-radius:8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------
# Robust path resolution
# ------------------------
try:
    SRC_FILE = Path(__file__).resolve()
    SRC_DIR = SRC_FILE.parent
    PROJECT_ROOT = SRC_DIR.parent
    # if not found as expected, fallback to cwd heuristics
    if not (PROJECT_ROOT / "data").exists() or not (PROJECT_ROOT / "models").exists():
        cwd = Path.cwd().resolve()
        if (cwd / "data").exists() and (cwd / "models").exists():
            PROJECT_ROOT = cwd
except NameError:
    SRC_DIR = Path.cwd().resolve()
    PROJECT_ROOT = SRC_DIR.parent if (SRC_DIR.parent / "data").exists() or (SRC_DIR.parent / "models").exists() else SRC_DIR

DATA_PATH = PROJECT_ROOT / "data" / "hotel_bookings.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "hotel_cancellation_model.pkl"

# ------------------------
# Loaders & helpers
# ------------------------
@st.cache_data(show_spinner=True)
def load_data(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Data file not found at: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    # numeric conversions & cleaning
    if "children" in df.columns:
        df["children"] = pd.to_numeric(df["children"], errors="coerce").fillna(0).astype(int)
    if "adr" in df.columns:
        df["adr"] = pd.to_numeric(df["adr"], errors="coerce").fillna(0.0)
        df.loc[df["adr"] < 0, "adr"] = 0.0
    if "is_canceled" in df.columns:
        df["is_canceled"] = pd.to_numeric(df["is_canceled"], errors="coerce").fillna(0).astype(int)

    # categorical normalization: convert to str and fill empty/NaN with 'Unknown'
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()
        df.loc[df[col] == "", col] = "Unknown"

    # country sometimes a float (NaN) -> ensure string
    if "country" in df.columns:
        df["country"] = df["country"].fillna("Unknown").astype(str).str.strip()
        df.loc[df["country"] == "", "country"] = "Unknown"

    # arrival_date_month ensure string
    if "arrival_date_month" in df.columns:
        df["arrival_date_month"] = df["arrival_date_month"].astype(str)

    return df


@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = joblib.load(path)
    return model


def safe_sorted(series):
    vals = pd.Series(series.dropna().astype(str).unique()).tolist()
    vals = [v for v in vals if v.strip() != ""]
    try:
        return sorted(vals)
    except Exception:
        return sorted(map(str, vals))


# ------------------------
# Plot helpers (used in EDA)
# ------------------------
def plot_count_by(df, column, top_n=15, title=None, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    if column not in df.columns:
        ax.text(0.5, 0.5, f"{column} missing", ha="center")
        ax.axis("off")
        return fig
    order = df[column].value_counts().nlargest(top_n).index
    sns.countplot(data=df, y=column, order=order, ax=ax)
    ax.set_title(title or f"Count by {column}")
    plt.tight_layout()
    return fig


def plot_cancellation_rate_by(df, column, title=None, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    if column not in df.columns or "is_canceled" not in df.columns:
        ax.text(0.5, 0.5, "Required column(s) missing", ha="center")
        ax.axis("off")
        return fig
    grp = df.groupby(column).agg(total=("is_canceled", "size"), canceled=("is_canceled", "sum"))
    grp = grp[grp["total"] > 0].copy()
    grp["cancel_rate"] = grp["canceled"] / grp["total"]
    grp = grp.sort_values("cancel_rate", ascending=False).head(30)
    grp["cancel_rate"].plot(kind="barh", ax=ax)
    ax.set_xlabel("Cancellation rate")
    ax.set_title(title or f"Cancellation rate by {column}")
    plt.tight_layout()
    return fig


def plot_corr_map(df, numeric_only=True, figsize=(10, 8)):
    if numeric_only:
        d = df.select_dtypes(include=[np.number])
    else:
        d = df.copy()
    corr = d.corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, cmap="vlag", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric features)")
    plt.tight_layout()
    return fig


def plot_lead_time_trend(df, by="arrival_date_month"):
    fig, ax = plt.subplots(figsize=(10, 4))
    if "lead_time" not in df.columns or by not in df.columns or "is_canceled" not in df.columns:
        ax.text(0.5, 0.5, "Required columns missing for this visualization", ha="center")
        ax.axis("off")
        return fig
    df = df.copy()
    df["lead_time_bin"] = pd.cut(df["lead_time"], bins=[-1, 7, 14, 30, 60, 90, 180, 365, 2000],
                                 labels=["0-7", "8-14", "15-30", "31-60", "61-90", "91-180", "181-365", "365+"])
    pivot = df.groupby([by, "lead_time_bin"]).agg(total=("is_canceled", "size"), canceled=("is_canceled", "sum")).reset_index()
    pivot["cancel_rate"] = pivot["canceled"] / pivot["total"]
    heat = pivot.pivot(index="lead_time_bin", columns=by, values="cancel_rate").fillna(0)
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax)
    ax.set_title("Cancellation rate by Lead Time bin and " + by)
    plt.tight_layout()
    return fig


# ------------------------
# Page: Home
# ------------------------
def home_page(df):
    st.markdown("<div class='title'>Hotel Booking Cancellation — IDS f24</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Interactive app: EDA, model, prediction and key takeaways</div>", unsafe_allow_html=True)
    st.write("---")

    left, right = st.columns([2, 1])
    with left:
        st.header("Project overview")
        st.markdown(
            """
            **Dataset:** Hotel bookings with attributes and cancellation flag.  
            **Project goal:** Predict cancellations to inform operational and revenue decisions.
            """
        )

        st.subheader("Quick stats")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{df.shape[0]:,}")
        c2.metric("Columns", f"{df.shape[1]}")
        c3.metric("Unique countries", f"{df['country'].nunique()}" if "country" in df.columns else "N/A")
        c4.metric("Cancellation rate", f"{df['is_canceled'].mean():.2%}" if "is_canceled" in df.columns else "N/A")
        st.write("### Sample data")
        st.dataframe(df.head(8))

    with right:
        st.header("Quick visuals")
        if "country" in df.columns:
            st.pyplot(plot_count_by(df, "country", top_n=8, figsize=(6, 3)))
        if "adr" in df.columns:
            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(df["adr"].clip(upper=df["adr"].quantile(0.99)), kde=True, ax=ax)
            ax.set_title("ADR distribution (clipped 99th pct)")
            st.pyplot(fig)

    st.write("---")


# ------------------------
# Page: EDA
# ------------------------
def eda_page(df):
    st.title("Exploratory Data Analysis")
    st.markdown("Pick an analysis and use the filters in the sidebar to slice the data.")

    st.sidebar.header("EDA Controls")
    choice = st.sidebar.selectbox("Analysis", ["Overview", "Lead Time Trends", "Market Segments", "Correlation", "Booking & ADR", "Detailed slices"])

    hotels_opt = safe_sorted(df["hotel"]) if "hotel" in df.columns else []
    hotel_filter = st.sidebar.multiselect("Hotel", options=hotels_opt, default=hotels_opt)

    months_opt = safe_sorted(df["arrival_date_month"]) if "arrival_date_month" in df.columns else []
    month_filter = st.sidebar.multiselect("Arrival month", options=months_opt, default=[])

    countries_opt = safe_sorted(df["country"]) if "country" in df.columns else []
    country_filter = st.sidebar.multiselect("Country", options=countries_opt, default=[])

    df_eda = df.copy()
    if hotel_filter:
        df_eda = df_eda[df_eda["hotel"].isin(hotel_filter)]
    if month_filter:
        df_eda = df_eda[df_eda["arrival_date_month"].isin(month_filter)]
    if country_filter:
        df_eda = df_eda[df_eda["country"].isin(country_filter)]

    if choice == "Overview":
        st.header("Overview")
        st.write("Aggregated summaries and distributions")
        st.subheader("Bookings by market segment")
        if "market_segment" in df_eda.columns:
            st.pyplot(plot_count_by(df_eda, "market_segment", top_n=12, figsize=(10, 4)))
        if "hotel" in df_eda.columns:
            st.subheader("Cancellation rate by hotel")
            st.pyplot(plot_cancellation_rate_by(df_eda, "hotel", figsize=(6, 3)))

    elif choice == "Lead Time Trends":
        st.header("Lead Time Trends")
        if "lead_time" in df_eda.columns:
            fig = plot_lead_time_trend(df_eda, by="arrival_date_month" if "arrival_date_month" in df_eda.columns else df_eda.columns[0])
            st.pyplot(fig)
            if "adr" in df_eda.columns and "is_canceled" in df_eda.columns:
                st.subheader("ADR vs Lead Time (sampled)")
                sample = df_eda.sample(frac=min(1.0, 10000 / max(1, df_eda.shape[0])), random_state=42)
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.scatterplot(data=sample, x="lead_time", y="adr", hue="is_canceled", alpha=0.6, ax=ax)
                ax.set_title("ADR vs Lead Time (colored by cancellation)")
                st.pyplot(fig)
        else:
            st.warning("lead_time not available in dataset.")

    elif choice == "Market Segments":
        st.header("Market Segments & Channels")
        if "market_segment" in df_eda.columns:
            st.pyplot(plot_count_by(df_eda, "market_segment", top_n=12, figsize=(10, 4)))
            st.pyplot(plot_cancellation_rate_by(df_eda, "market_segment", figsize=(8, 4)))
        if "distribution_channel" in df_eda.columns and "customer_type" in df_eda.columns:
            st.subheader("Distribution Channel vs Customer Type")
            pivot = pd.crosstab(df_eda["distribution_channel"], df_eda["customer_type"], normalize="index")
            fig, ax = plt.subplots(figsize=(10, 4))
            pivot.plot(kind="bar", stacked=True, ax=ax)
            ax.set_ylabel("Proportion")
            ax.set_title("Distribution Channel vs Customer Type")
            st.pyplot(fig)

    elif choice == "Correlation":
        st.header("Correlation matrix")
        st.pyplot(plot_corr_map(df_eda, numeric_only=True, figsize=(12, 8)))
        if "is_canceled" in df_eda.columns:
            st.markdown("Top correlations with cancellation:")
            corr = df_eda.select_dtypes(include=[np.number]).corr()["is_canceled"].drop("is_canceled").sort_values()
            st.write(pd.DataFrame({"feature": corr.index, "corr_with_canceled": corr.values}).head(8))
            st.write(pd.DataFrame({"feature": corr.index, "corr_with_canceled": corr.values}).tail(8))

    elif choice == "Booking & ADR":
        st.header("Booking & ADR analysis")
        if "adr" in df_eda.columns and "hotel" in df_eda.columns:
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.boxplot(data=df_eda[df_eda["adr"] < df_eda["adr"].quantile(0.99)], x="hotel", y="adr", ax=ax)
            ax.set_title("ADR by Hotel (clipped 99th pct)")
            st.pyplot(fig)
        if "adr" in df_eda.columns and "is_canceled" in df_eda.columns:
            st.subheader("Cancellation by ADR decile")
            df_eda["adr_decile"] = pd.qcut(df_eda["adr"].replace(0, np.nan).fillna(0) + 1, 10, labels=False)
            decile = df_eda.groupby("adr_decile").agg(cancel_rate=("is_canceled", "mean"), avg_adr=("adr", "mean")).reset_index()
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.lineplot(data=decile, x="avg_adr", y="cancel_rate", marker="o", ax=ax)
            ax.set_xlabel("Average ADR by decile")
            ax.set_ylabel("Cancellation rate")
            st.pyplot(fig)

    else:
        st.header("Detailed slices")
        st.markdown("Targeted micro-analyses useful for project writeups.")
        if "is_repeated_guest" in df_eda.columns and "is_canceled" in df_eda.columns:
            fig, ax = plt.subplots(figsize=(6, 3))
            s = df_eda.groupby("is_repeated_guest").is_canceled.mean()
            s.plot(kind="bar", ax=ax)
            ax.set_title("Cancellation rate: repeated guest vs not")
            st.pyplot(fig)


# ------------------------
# Page: Model (clean) with sliders + row input preview
# ------------------------
def model_clean_page(df):
    """
    Minimal, clean model UI:
      - Technical bullets about model
      - Sliders & selects for prediction inputs
      - Input preview shows row
      - Clean output: probability (%) and label
    """
    st.title("Predictor — Clean Input & Clean Output")
    st.markdown("Clean technical bullets and a tidy slider-based input form. Output shows cancellation probability (%) and label.")

    # Load model
    try:
        model = load_model(str(MODEL_PATH))
        st.success("Saved model loaded.")
    except Exception as e:
        st.error("Could not load model: " + str(e))
        return

    # Model bullets (concise)
    st.subheader("Model summary:")
    try:
        model_type = type(model).__name__
        pipeline_steps = getattr(model, "named_steps", None)
        last_est = None
        if pipeline_steps:
            steps_list = list(pipeline_steps.keys())
            st.markdown(f"- Pipeline steps: `{ ' -> '.join(steps_list) }`")
            last_est = list(pipeline_steps.items())[-1][1]
        else:
            st.markdown("- Pipeline: _not detected_ (model saved may be a raw estimator)")
        if last_est is not None:
            est_name = type(last_est).__name__
        else:
            est_name = model_type
        st.markdown(
            f"- Model type: **{est_name}** (saved object: `{model_type}`)\n"
            "- Algorithm used during training: **Random Forest**.\n"
            "- The pipeline should handle preprocessing: encoding, imputation, and scaling."
        )
    except Exception:
        st.markdown("- Model metadata not available (saved object may be custom).")

    st.write("---")

    # Determine expected feature names if available
    if hasattr(model, "feature_names_in_"):
        expected_cols = list(model.feature_names_in_)
    else:
        expected_cols = None
        if hasattr(model, "named_steps"):
            last = list(model.named_steps.items())[-1][1]
            if hasattr(last, "feature_names_in_"):
                expected_cols = list(last.feature_names_in_)
        if expected_cols is None:
            expected_cols = [c for c in df.columns if c != "is_canceled"]

    # Sliders & selects for core features (clean UI)
    st.subheader("Enter booking details:")

    # Determine reasonable slider ranges based on dataset
    max_lead = int(min(1000, df["lead_time"].max())) if "lead_time" in df.columns else 365
    max_adr = float(min(10000.0, df["adr"].quantile(0.99))) if "adr" in df.columns else 1000.0
    max_prev_canc = int(min(50, df["previous_cancellations"].max())) if "previous_cancellations" in df.columns else 10
    max_adults = int(min(10, df["adults"].max())) if "adults" in df.columns else 5
    max_children = int(min(10, df["children"].max())) if "children" in df.columns else 5

    with st.form("clean_predict_form_sliders"):
        c1, c2, c3 = st.columns([1, 1, 1])

        with c1:
            lead_time = st.slider("Lead time (days)", min_value=0, max_value=max_lead, value=min(30, max_lead))
            adults = st.slider("Adults", min_value=0, max_value=max_adults, value=2)
            children = st.slider("Children", min_value=0, max_value=max_children, value=0)
            booking_changes = st.slider("Booking changes", min_value=0, max_value=20, value=0)

        with c2:
            adr_val = float(df["adr"].median()) if "adr" in df.columns else 100.0
            adr = st.slider("ADR (avg daily rate)", min_value=0.0, max_value=max_adr, value=float(adr_val), step=1.0)
            prev_canc = st.slider("Previous cancellations", min_value=0, max_value=max_prev_canc, value=0)
            total_special = st.slider("Total special requests", min_value=0, max_value=20, value=0)

        with c3:
            hotel_opt = safe_sorted(df["hotel"]) if "hotel" in df.columns else ["City Hotel", "Resort Hotel"]
            hotel = st.selectbox("Hotel", options=hotel_opt, index=0)
            deposit_opt = safe_sorted(df["deposit_type"]) if "deposit_type" in df.columns else ["No Deposit"]
            deposit_type = st.selectbox("Deposit type", options=deposit_opt, index=0)
            country_opt = safe_sorted(df["country"]) if "country" in df.columns else ["Unknown"]
            country = st.selectbox("Country", options=country_opt, index=0)

        submit = st.form_submit_button("Predict (%)")

    # On submit, align inputs to expected_cols and predict
    if submit:
        input_row = {}
        collected = {
            "lead_time": int(lead_time),
            "adults": int(adults),
            "children": int(children),
            "adr": float(adr),
            "previous_cancellations": int(prev_canc),
            "hotel": hotel,
            "deposit_type": deposit_type,
            "country": country,
            "booking_changes": int(booking_changes),
            "total_of_special_requests": int(total_special),
        }

        for col in expected_cols:
            if col in collected:
                input_row[col] = collected[col]
            else:
                if col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        input_row[col] = float(df[col].median())
                    else:
                        mode = df[col].mode()
                        input_row[col] = str(mode.iloc[0]) if (not mode.empty) else "Unknown"
                else:
                    # safe fallback
                    input_row[col] = 0 if col.lower().endswith(("id", "num")) else ""

        input_df = pd.DataFrame([input_row])

        st.markdown("**Input preview**")
        # Show as rows (not transposed)
        st.dataframe(input_df)

        # Predict probability (clean)
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(input_df)[:, 1][0]
            else:
                pred = model.predict(input_df)[0]
                prob = float(pred) if isinstance(pred, (int, float)) else (1.0 if pred == 1 else 0.0)

            pct = prob * 100.0
            label = "Canceled" if prob >= 0.5 else "Check-In"

            st.markdown("### Prediction")
            st.metric(label="Probability of Cancellation", value=f"{pct:.1f}%")
            st.markdown(f"**Predicted label:**")
            if label == "Canceled":
                st.markdown(
                    f"<span style='color:#b42318; font-size:22px; font-weight:600;'>CANCELED</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<span style='color:#1e7f43; font-size:22px; font-weight:600;'>CHECK-IN</span>",
                    unsafe_allow_html=True
                )         
   
            # show progress bar as a visual indicator
            st.progress(min(max(int(round(pct)), 0), 100))

        except Exception as e:
            st.error("Prediction failed: " + str(e))
            st.info("If this happens, the saved pipeline may expect a different schema/dtypes. Ensure the training pipeline included preprocessing and was saved with the model.")

# ------------------------
# Page: Key Takeaways
# ------------------------
def key_takeaways_page(df):
    st.title("Key Takeaways")
    st.markdown("Concise findings from the EDA & modeling.")

    st.subheader("Top insights")
    st.markdown(
        """
        - **Seasonality**: arrival month strongly influences booking volume and cancellation patterns.
        - **Lead time**: long lead times are an important predictive signal for cancellations.
        - **Revenue signals**: ADR interacts with cancellation probability — observe non-linear relationships.
        - **Channel & market differences**: cancellation rates differ by market_segment and distribution_channel.
        - **Behavioral predictors**: previous_cancellations and is_repeated_guest provide strong behavioral signals.
        - **Model caveat**: predictions require matching preprocessing/schema between training and inference.
        """
    )

    st.subheader("Practical recommendations")
    st.markdown(
        """
        - Target likely cancellations with reminders or retention offers; offer upsells to likely check-ins.  
        - Consider deposit or stricter policies for high-risk booking profiles.  
        - Retrain periodically to address concept drift as booking behaviors evolve.  
        - Validate operational changes (A/B testing) before full rollout.
        """
    )

# ------------------------
# App navigation
# ------------------------
def main():
    # Load data (fail gracefully)
    try:
        df = load_data(str(DATA_PATH))
    except Exception as e:
        st.error(f"Could not load data from {DATA_PATH}: {e}")
        st.stop()

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "EDA", "Predictor", "Key Takeaways"], index=0)

    with st.sidebar.expander("Project files / info"):
        st.markdown(f"- Dataset: `../data/hotel_bookings.csv`\n- Model: `../models/hotel_cancellation_model.pkl`\n- App: `src/app.py`")
        st.markdown("About: This UI is prepared to present EDA, a clean predictor, and concise takeaways for IDS f24.")
        st.markdown('Access code on [GitHub](https://github.com/asqasim/hotel-booking-project)')
        st.markdown(
            """
        **Made By:** Ahmad Sohaib Qasim  
        Data Analyst  
        [Website](https://asqasim.netlify.app)
        """
        )



    if page == "Home":
        home_page(df)
    elif page == "EDA":
        eda_page(df)
    elif page == "Predictor":
        model_clean_page(df)
    elif page == "Key Takeaways":
        key_takeaways_page(df)
    else:
        st.write("Page not found.")

    st.write("---")
    st.markdown("<small class='small-muted'>Built for IDS f24 — ensure saved pipeline includes preprocessing so the Predictor operates correctly.</small>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
