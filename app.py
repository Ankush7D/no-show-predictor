# =============================================================================
#  Patient No-Show Predictor — Cevi AI Internship Project
#  -------------------------------------------------------
#  HOW THE APP WORKS (read this first!):
#
#  1. You upload a CSV with appointment records (or use built-in demo data).
#  2. You tell the app which column is your TARGET (did the patient no-show?).
#  3. The app automatically uses ALL OTHER columns as input features.
#  4. A Random Forest model is trained and you get:
#       - Charts exploring the data
#       - Model accuracy metrics
#       - A live predictor where you enter one patient's details
#
#  HOW TO RUN:
#    pip install streamlit pandas numpy matplotlib seaborn scikit-learn
#    streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
#  CONFIGURATION
#  Change values here to adjust the app's behaviour.
#  No need to touch anything else in the file.
# =============================================================================

CONFIG = {
    "app_title"          : "No-Show Predictor",
    "app_icon"           : "🏥",
    "cost_per_slot"      : 180,   # dollars lost when a patient skips their slot
    "high_risk_threshold": 60,    # % → HIGH risk badge
    "medium_risk_threshold": 35,  # % → MEDIUM risk badge (below this = LOW)
    "test_split"         : 0.20,  # fraction of data kept for evaluation
    "random_seed"        : 42,
    "n_trees"            : 200,   # number of trees in the Random Forest
    "max_tree_depth"     : 8,
}

# =============================================================================
#  INTERVENTION RULES
#  -------------------------------------------------------
#  Each rule is a dict with two keys:
#    "condition" — a function that receives the patient's data as a dict
#                  and returns True / False
#    "message"   — what Cevi's agent should do when this rule fires
#
#  To ADD a rule  → add a new dict to the list below.
#  To REMOVE one  → delete (or comment out) its dict.
#  No code changes needed anywhere else.
# =============================================================================

INTERVENTION_RULES = [
    {
        "condition": lambda p: p.get("sms_sent", 1) == 0,
        "message"  : "📱 No SMS sent — send a reminder immediately (reduces no-shows by ~12%)",
    },
    {
        "condition": lambda p: p.get("lead_days", 0) >= 14,
        "message"  : "📞 Long lead time — schedule a confirmation call 48 hours before",
    },
    {
        "condition": lambda p: p.get("prev_noshows", 0) >= 2,
        "message"  : "🔁 Repeat no-shower — consider double-booking or adding to waitlist",
    },
    {
        "condition": lambda p: p.get("appt_hour", 10) >= 15,
        "message"  : "⏰ Late afternoon slot — offer an earlier time (afternoon = ~18% riskier)",
    },
    {
        "condition": lambda p: p.get("scholarship", 0) == 1,
        "message"  : "🤝 Welfare patient — check transport access or offer telehealth",
    },
    {
        "condition": lambda p: p.get("hypertension", 0) == 1 or p.get("diabetes", 0) == 1,
        "message"  : "💊 Chronic condition — add a relevant health tip to the outreach message",
    },
    {
        "condition": lambda p: p.get("__risk_pct__", 0) >= CONFIG["high_risk_threshold"],
        "message"  : "🚨 HIGH risk — trigger a voice call today; do not rely on SMS alone",
    },
]


# =============================================================================
#  DEMO DATA GENERATOR
#  -------------------------------------------------------
#  Only used when the user has not uploaded their own CSV.
#  The formula inside mirrors real no-show patterns from the
#  Kaggle "Medical Appointment No-Shows" dataset.
# =============================================================================

@st.cache_data
def generate_demo_data(n_rows: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Creates a realistic synthetic dataset of appointment records."""
    rng = np.random.default_rng(seed)

    age          = rng.integers(0, 95, n_rows)
    gender       = rng.choice(["F", "M"], n_rows, p=[0.65, 0.35])
    lead_days    = rng.integers(0, 60, n_rows)
    sms_sent     = rng.choice([0, 1], n_rows, p=[0.32, 0.68])
    hypertension = rng.choice([0, 1], n_rows, p=[0.80, 0.20])
    diabetes     = rng.choice([0, 1], n_rows, p=[0.92, 0.08])
    alcoholism   = rng.choice([0, 1], n_rows, p=[0.97, 0.03])
    scholarship  = rng.choice([0, 1], n_rows, p=[0.90, 0.10])
    appt_hour    = rng.integers(7, 18, n_rows)
    prev_noshows = rng.integers(0, 6, n_rows)

    # Logistic formula: positive coefficient = raises no-show risk
    logit = (
        -0.80
        + 0.008 * lead_days
        - 0.005 * age
        + 0.200 * (gender == "M")
        - 0.550 * sms_sent
        + 0.350 * scholarship
        + 0.500 * prev_noshows
        - 0.150 * hypertension
        + 0.180 * (appt_hour >= 15)
        + rng.normal(0, 0.4, n_rows)   # random noise
    )
    prob    = 1 / (1 + np.exp(-logit))
    no_show = (rng.random(n_rows) < prob).astype(int)

    return pd.DataFrame({
        "age": age, "gender": gender, "lead_days": lead_days,
        "sms_sent": sms_sent, "hypertension": hypertension,
        "diabetes": diabetes, "alcoholism": alcoholism,
        "scholarship": scholarship, "appt_hour": appt_hour,
        "prev_noshows": prev_noshows, "no_show": no_show,
    })


# =============================================================================
#  AUTO PREPROCESSING
#  -------------------------------------------------------
#  This function takes ANY dataframe and automatically:
#    - Drops columns that are mostly empty (>50% missing)
#    - Encodes text columns into numbers (LabelEncoder)
#    - Fills remaining missing values with the column median
#
#  No hardcoded column names — it works on whatever CSV you upload.
# =============================================================================

def auto_preprocess(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    """
    Automatically prepares a DataFrame for ML.

    Returns:
        X — feature matrix (everything except the target)
        y — target series (the no-show column)
    """
    data = df.copy()

    # ── Step 1: drop columns that are more than 50% empty ──────────────
    threshold = 0.50
    missing_fraction = data.isnull().mean()
    cols_to_drop = missing_fraction[missing_fraction > threshold].index.tolist()
    if cols_to_drop:
        data.drop(columns=cols_to_drop, inplace=True)

    # ── Step 2: separate features (X) and target (y) ───────────────────
    y = pd.to_numeric(data[target_col], errors="coerce").fillna(0).astype(int)
    X = data.drop(columns=[target_col])

    # ── Step 3: encode text columns automatically ───────────────────────
    # LabelEncoder turns e.g. ["F", "M"] into [0, 1]
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    # ── Step 4: fill any remaining missing values with the column median ─
    X = X.fillna(X.median(numeric_only=True))

    # ── Step 5: keep only numeric columns (safety net) ──────────────────
    X = X.select_dtypes(include="number")

    return X, y


# =============================================================================
#  MODEL TRAINING
#  -------------------------------------------------------
#  @st.cache_resource means Streamlit only retrains when the data changes.
#  The cache_key argument is a fingerprint of the data; if it changes,
#  the model retrains automatically.
# =============================================================================

@st.cache_resource
def train_model(cache_key: str, X: pd.DataFrame, y: pd.Series):
    """
    Splits data 80/20, trains a Random Forest, returns everything needed
    to evaluate and display results.

    Returns:
        model    — the trained classifier
        X_test   — held-out feature rows
        y_test   — held-out labels
        features — list of feature column names (in model order)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size   = CONFIG["test_split"],
        random_state= CONFIG["random_seed"],
        stratify    = y,        # keep class ratio equal in train & test
    )

    model = RandomForestClassifier(
        n_estimators  = CONFIG["n_trees"],
        max_depth     = CONFIG["max_tree_depth"],
        min_samples_leaf = 10,
        class_weight  = "balanced",     # handles unequal show/no-show counts
        random_state  = CONFIG["random_seed"],
        n_jobs        = -1,             # use all CPU cores
    )
    model.fit(X_train, y_train)

    return model, X_test, y_test, list(X.columns)


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

COLORS = {
    "green": "#059669",
    "red"  : "#ef4444",
    "amber": "#f59e0b",
    "blue" : "#3b82f6",
    "gray" : "#9ca3af",
    "light": "#f3f4f6",
}

def white_fig(w: float = 5, h: float = 3.5):
    """Returns a (fig, ax) pair with a clean white background."""
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    return fig, ax

def add_bar_labels(ax, bars, fmt=".1f", suffix="%"):
    """Adds a value label on top of each bar in a bar chart."""
    for bar in bars:
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.3,
            f"{h:{fmt}}{suffix}", ha="center",
            fontsize=10, fontweight="bold", color="#111827"
        )

def stat_card(col, value: str, label: str):
    """Renders a green KPI card inside a Streamlit column."""
    col.markdown(f"""
    <div style='background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                padding:18px 20px;text-align:center;'>
        <p style='font-size:1.9rem;font-weight:800;color:#059669;margin:0;'>{value}</p>
        <p style='font-size:0.75rem;color:#6b7280;text-transform:uppercase;
                  letter-spacing:0.8px;margin-top:4px;'>{label}</p>
    </div>""", unsafe_allow_html=True)

def action_card(text: str):
    """Renders a blue-bordered recommendation card."""
    st.markdown(f"""
    <div style='background:#f9fafb;border-left:3px solid #3b82f6;
                border-radius:0 6px 6px 0;padding:10px 14px;
                margin:6px 0;font-size:0.88rem;color:#1f2937;'>
        {text}
    </div>""", unsafe_allow_html=True)


# =============================================================================
#  STREAMLIT APP STARTS HERE
# =============================================================================

st.set_page_config(
    page_title=CONFIG["app_title"],
    page_icon=CONFIG["app_icon"],
    layout="wide",
)

# Global CSS — white background, clean typography
st.markdown("""
<style>
[data-testid="stAppViewContainer"], .main { background: #ffffff; }
[data-testid="stSidebar"] {
    background: #f8f9fb;
    border-right: 1px solid #e5e7eb;
}
[data-testid="metric-container"] { display: none; }
h1 { color: #111827; font-size: 1.8rem !important; font-weight: 700 !important; }
h2, h3 { color: #1f2937; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
#  SIDEBAR — navigation and settings
# =============================================================================

with st.sidebar:
    st.markdown(f"### {CONFIG['app_icon']} {CONFIG['app_title']}")
    st.divider()

    page = st.radio("Navigation", [
        "📂 Data Setup",
        "📊 Data Explorer",
        "🤖 Model Results",
        "🎯 Live Predictor",
    ])

    st.divider()

    # Let the user adjust the revenue assumption without touching any code
    cost_per_slot = st.number_input(
        "Revenue per appointment slot ($)",
        value=CONFIG["cost_per_slot"],
        min_value=1,
        help="Used to estimate revenue recovered when an intervention works",
    )


# =============================================================================
#  PAGE 0 — DATA SETUP
#  -------------------------------------------------------
#  The user uploads their CSV (or uses demo data).
#  They only need to pick ONE thing: which column is the target.
#  All other columns are automatically used as features.
# =============================================================================

if page == "📂 Data Setup":

    st.title("Step 1 — Load Your Data")
    st.markdown("""
    Upload **any CSV** with appointment records.

    - You only need to tell the app **which column is the target**
      (did the patient no-show? — usually `0` = showed up, `1` = missed).
    - **Every other column** is automatically used as a feature.
    - Text columns (like gender) are encoded to numbers automatically.
    - No manual feature selection needed.
    """)

    st.divider()

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file:
        # Load the raw CSV
        raw_df = pd.read_csv(uploaded_file)
        st.success(f"✅ File loaded: **{len(raw_df):,} rows** × **{len(raw_df.columns)} columns**")

        with st.expander("Preview first 5 rows"):
            st.dataframe(raw_df.head(), use_container_width=True)

        st.divider()
        st.subheader("Pick Your Target Column")
        st.markdown(
            "Select the column that says whether the patient **missed** their appointment. "
            "It should contain `1` (missed) and `0` (showed up)."
        )

        target_col = st.selectbox(
            "Target column (no-show label)",
            options=raw_df.columns.tolist(),
            # Try to pre-select a column whose name contains "no_show" or "noshow"
            index=next(
                (i for i, c in enumerate(raw_df.columns)
                 if "no" in c.lower() and "show" in c.lower()),
                0
            ),
        )

        # Show a quick value breakdown so the user can verify
        st.caption(f"Value counts for **{target_col}**:")
        st.write(raw_df[target_col].value_counts().to_frame("count"))

        if st.button("✅ Confirm & Build Model", use_container_width=True):
            # Preprocess and store in session so all pages can use it
            X, y = auto_preprocess(raw_df, target_col)

            st.session_state["X"]          = X
            st.session_state["y"]          = y
            st.session_state["raw_df"]     = raw_df
            st.session_state["target_col"] = target_col
            st.session_state["is_demo"]    = False

            # Show which columns were auto-selected as features
            st.success(
                f"✅ Model will use **{len(X.columns)} features** automatically. "
                "Navigate using the sidebar."
            )
            st.markdown("**Features selected automatically:**")
            st.write(list(X.columns))

    else:
        # ── Demo data path ────────────────────────────────────────────────
        st.info(
            "No file yet — you can use the built-in demo data below. "
            "It mirrors the real Kaggle Medical Appointment No-Shows dataset."
        )
        n_demo = st.slider("How many demo rows?", 500, 10_000, 3_000, 500)

        if st.button("▶ Load Demo Data", use_container_width=True):
            demo_df = generate_demo_data(n_demo, CONFIG["random_seed"])
            target_col = "no_show"

            X, y = auto_preprocess(demo_df, target_col)

            st.session_state["X"]          = X
            st.session_state["y"]          = y
            st.session_state["raw_df"]     = demo_df
            st.session_state["target_col"] = target_col
            st.session_state["is_demo"]    = True

            st.success(
                f"✅ Demo data loaded — {len(X.columns)} features, {len(y):,} rows. "
                "Navigate using the sidebar."
            )


# =============================================================================
#  GUARD — pages 2, 3, 4 need data first
# =============================================================================

elif "X" not in st.session_state:
    st.title(f"{CONFIG['app_icon']} {CONFIG['app_title']}")
    st.warning("👈 Go to **Data Setup** first and load your data.")
    st.stop()

# =============================================================================
#  SHARED SETUP — runs for all pages after data is loaded
# =============================================================================

else:
    # Pull data from session state
    X          = st.session_state["X"]
    y          = st.session_state["y"]
    raw_df     = st.session_state["raw_df"]
    target_col = st.session_state["target_col"]
    is_demo    = st.session_state.get("is_demo", False)

    # Build a cache key so the model retrains automatically if data changes
    cache_key = str(pd.util.hash_pandas_object(X.head(500)).sum())
    model, X_test, y_test, feature_names = train_model(cache_key, X, y)

    # Summary stats used across pages
    no_show_rate    = y.mean()
    n_noshows       = int(y.sum())
    revenue_at_risk = n_noshows * cost_per_slot
    data_label      = "Demo Data" if is_demo else "Your Data"

    # =========================================================================
    #  PAGE 1 — DATA EXPLORER
    #  -------------------------------------------------------
    #  Shows charts for columns that are present in the data.
    #  If a column is missing, its chart is simply skipped.
    # =========================================================================

    if page == "📊 Data Explorer":

        st.title(f"Data Explorer — {data_label}")
        st.markdown("Explore patterns in the data before looking at model results.")

        # ── KPI cards ────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        stat_card(c1, f"{len(y):,}",              "Total Appointments")
        stat_card(c2, f"{no_show_rate*100:.1f}%", "No-Show Rate")
        stat_card(c3, f"{n_noshows:,}",           "Missed Slots")
        stat_card(c4, f"${revenue_at_risk:,.0f}", "Revenue at Risk")

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Utility: group a column into buckets and plot no-show rate ────
        def plot_noshow_by_group(col_name, bins, bin_labels, x_label, chart_type="bar"):
            """
            Generic chart: group `col_name` into `bins` and plot no-show rate.
            Works for any numeric column — no column-name hardcoding.
            """
            series = pd.to_numeric(raw_df[col_name], errors="coerce")
            group  = pd.cut(series, bins=bins, labels=bin_labels, include_lowest=True).fillna(bin_labels[-1])
            rates  = raw_df.groupby(group, observed=True)[target_col].apply(
                lambda s: pd.to_numeric(s, errors="coerce").mean() * 100
            ).dropna()

            fig, ax = white_fig()
            if chart_type == "line":
                ax.plot(rates.index.astype(str), rates.values,
                        color=COLORS["green"], marker="o", linewidth=2.5, markersize=8, zorder=3)
                ax.fill_between(range(len(rates)), rates.values, alpha=0.12, color=COLORS["green"])
                ax.set_ylim(0, rates.max() * 1.3)
                ax.grid(linestyle="--", alpha=0.35)
            else:
                bars = ax.bar(rates.index.astype(str), rates.values,
                              color=COLORS["amber"], edgecolor="none", zorder=3)
                ax.set_ylim(0, rates.max() * 1.3)
            ax.set_xlabel(x_label)
            ax.set_ylabel("No-Show Rate (%)")
            return fig

        # ── Row 1: Gender (if present) | Age (if present) ────────────────
        col_a, col_b = st.columns(2)

        with col_a:
            if "gender" in raw_df.columns:
                st.subheader("No-Show Rate by Gender")
                gender_rates = raw_df.groupby("gender")[target_col].apply(
                    lambda s: pd.to_numeric(s, errors="coerce").mean() * 100
                )
                fig, ax = white_fig()
                bars = ax.bar(gender_rates.index, gender_rates.values,
                              color=[COLORS["green"], COLORS["blue"]],
                              edgecolor="none", zorder=3)
                add_bar_labels(ax, bars)
                ax.set_ylabel("No-Show Rate (%)")
                ax.set_ylim(0, gender_rates.max() * 1.3)
                st.pyplot(fig); plt.close()
            else:
                st.info("No **gender** column found — chart skipped.")

        with col_b:
            if "age" in raw_df.columns:
                st.subheader("No-Show Rate by Age Group")
                fig = plot_noshow_by_group(
                    "age",
                    bins=[0, 17, 35, 55, 75, 999],
                    bin_labels=["0-17", "18-35", "36-55", "56-75", "75+"],
                    x_label="Age Group",
                    chart_type="line",
                )
                st.pyplot(fig); plt.close()
            else:
                st.info("No **age** column found — chart skipped.")

        # ── Row 2: Lead days (if present) | SMS (if present) ─────────────
        col_c, col_d = st.columns(2)

        with col_c:
            if "lead_days" in raw_df.columns:
                st.subheader("Lead Time vs No-Show Rate")
                fig = plot_noshow_by_group(
                    "lead_days",
                    bins=[0, 7, 14, 21, 30, 9999],
                    bin_labels=["0-7d", "8-14d", "15-21d", "22-30d", "30d+"],
                    x_label="Days Between Booking & Appointment",
                )
                st.pyplot(fig); plt.close()
            else:
                st.info("No **lead_days** column found — chart skipped.")

        with col_d:
            if "sms_sent" in raw_df.columns:
                st.subheader("SMS Reminder Impact")
                sms_num   = pd.to_numeric(raw_df["sms_sent"], errors="coerce").fillna(0).astype(int)
                sms_rates = raw_df.groupby(sms_num)[target_col].apply(
                    lambda s: pd.to_numeric(s, errors="coerce").mean() * 100
                )
                v_no  = float(sms_rates.get(0, 0))
                v_yes = float(sms_rates.get(1, 0))

                fig, ax = white_fig()
                bars = ax.bar(["No SMS", "SMS Sent"], [v_no, v_yes],
                              color=[COLORS["red"], COLORS["green"]],
                              edgecolor="none", zorder=3)
                add_bar_labels(ax, bars)
                ax.set_title(f"SMS cuts no-shows by {v_no - v_yes:.1f} pp", fontsize=10)
                ax.set_ylabel("No-Show Rate (%)")
                ax.set_ylim(0, max(v_no, v_yes) * 1.3)
                st.pyplot(fig); plt.close()
            else:
                st.info("No **sms_sent** column found — chart skipped.")

        # ── Correlation chart (works on all numeric columns automatically) ─
        st.subheader("Which Features Correlate with No-Show?")
        st.caption("Positive = raises no-show risk  |  Negative = lowers it")

        all_numeric = X.copy()
        all_numeric[target_col] = y
        corr = all_numeric.corr()[target_col].drop(target_col).dropna().sort_values()
        bar_colors = [COLORS["red"] if v < 0 else COLORS["green"] for v in corr.values]

        fig, ax = plt.subplots(figsize=(10, max(3, len(corr) * 0.4)))
        ax.barh(corr.index, corr.values, color=bar_colors, edgecolor="none")
        ax.axvline(0, color=COLORS["gray"], linewidth=1)
        ax.set_xlabel("Pearson Correlation with No-Show")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        st.pyplot(fig); plt.close()

        # ── Raw data peek ─────────────────────────────────────────────────
        with st.expander("📋 View raw data sample"):
            st.dataframe(raw_df.sample(min(100, len(raw_df))), use_container_width=True)


    # =========================================================================
    #  PAGE 2 — MODEL RESULTS
    # =========================================================================

    elif page == "🤖 Model Results":

        st.title("Model Performance")

        train_pct = int((1 - CONFIG["test_split"]) * 100)
        test_pct  = int(CONFIG["test_split"] * 100)
        st.markdown(
            f"The Random Forest was trained on **{train_pct}%** of rows "
            f"and evaluated on the remaining **{test_pct}%** it never saw."
        )

        # ── Compute metrics ───────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        report = classification_report(y_test, y_pred, output_dict=True)
        auc    = roc_auc_score(y_test, y_prob)

        # 5-fold cross-validation gives a more reliable AUC estimate
        cv_scores = cross_val_score(model, X, y, cv=5, scoring="roc_auc")

        # ── KPI cards ────────────────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        stat_card(c1, f"{auc:.3f}",                                 "ROC-AUC")
        stat_card(c2, f"{report['1']['precision']:.3f}",            "Precision (No-Show)")
        stat_card(c3, f"{report['1']['recall']:.3f}",               "Recall (No-Show)")
        stat_card(c4, f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}", "5-Fold CV AUC")

        st.markdown("<br>", unsafe_allow_html=True)
        st.info(
            "**ROC-AUC** ranges from 0.5 (random guessing) to 1.0 (perfect). "
            "Above 0.70 is considered good for this kind of clinical prediction task."
        )

        col_a, col_b = st.columns(2)

        # ROC Curve
        with col_a:
            st.subheader("ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig, ax = white_fig()
            ax.plot(fpr, tpr, color=COLORS["green"], linewidth=2.5, label=f"AUC = {auc:.3f}")
            ax.fill_between(fpr, tpr, alpha=0.12, color=COLORS["green"])
            ax.plot([0, 1], [0, 1], "--", color=COLORS["gray"],
                    linewidth=1, label="Random baseline (AUC = 0.5)")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(linestyle="--", alpha=0.35)
            st.pyplot(fig); plt.close()

        # Confusion Matrix
        with col_b:
            st.subheader("Confusion Matrix")
            st.caption("Rows = actual labels, Columns = what the model predicted")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = white_fig()
            sns.heatmap(
                cm, annot=True, fmt="d", ax=ax,
                cmap=sns.light_palette(COLORS["green"], as_cmap=True),
                xticklabels=["Showed Up", "No-Show"],
                yticklabels=["Showed Up", "No-Show"],
                linewidths=2, linecolor="white",
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            fig.patch.set_facecolor("#ffffff")
            st.pyplot(fig); plt.close()

        # Feature Importance — works automatically for any features in the model
        st.subheader("Feature Importance")
        st.caption("Which columns does the model rely on most heavily?")

        importance = (
            pd.Series(model.feature_importances_, index=feature_names)
            .sort_values(ascending=True)
        )
        imp_colors = [
            COLORS["green"] if v > importance.median() else COLORS["blue"]
            for v in importance.values
        ]

        fig, ax = plt.subplots(figsize=(10, max(3, len(importance) * 0.45)))
        ax.barh(importance.index, importance.values, color=imp_colors, edgecolor="none")
        ax.set_xlabel("Importance Score")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        st.pyplot(fig); plt.close()
        st.caption("🟢 High importance  |  🔵 Moderate importance")


    # =========================================================================
    #  PAGE 3 — LIVE PREDICTOR
    #  -------------------------------------------------------
    #  The app automatically builds input widgets for every feature
    #  that exists in the model — no hardcoding of field names.
    # =========================================================================

    elif page == "🎯 Live Predictor":

        st.title("Live No-Show Risk Predictor")
        st.markdown(
            "Adjust the sliders and dropdowns for a single patient. "
            "The model gives an instant **risk percentage** and "
            "a list of actions for Cevi's agents."
        )

        col_form, col_result = st.columns([1, 1], gap="large")

        with col_form:
            st.subheader("Patient Details")

            # ── Auto-generate one input widget per feature ─────────────────
            # We look at the data to decide what kind of widget to show:
            #   • Binary (only 0 and 1) → Yes/No radio
            #   • A few unique values   → selectbox
            #   • Continuous numbers   → slider
            patient_input = {}

            for col_name in feature_names:
                col_data   = X[col_name].dropna()
                unique_vals = sorted(col_data.unique())
                col_min    = float(col_data.min())
                col_max    = float(col_data.max())
                col_median = float(col_data.median())

                # Binary feature (e.g. sms_sent: 0 or 1)
                if set(unique_vals).issubset({0, 1}):
                    val = st.radio(
                        col_name.replace("_", " ").title(),
                        options=[0, 1],
                        format_func=lambda x: "Yes" if x == 1 else "No",
                        horizontal=True,
                        key=f"pred_{col_name}",
                    )

                # Categorical with few options (≤ 10 unique values)
                elif len(unique_vals) <= 10:
                    val = st.selectbox(
                        col_name.replace("_", " ").title(),
                        options=unique_vals,
                        index=min(len(unique_vals) - 1,
                                  unique_vals.index(col_median)
                                  if col_median in unique_vals else 0),
                        key=f"pred_{col_name}",
                    )

                # Continuous numeric → slider
                else:
                    val = st.slider(
                        col_name.replace("_", " ").title(),
                        min_value=int(col_min),
                        max_value=int(col_max),
                        value=int(col_median),
                        key=f"pred_{col_name}",
                    )

                patient_input[col_name] = val

            predict_clicked = st.button("🔍 Predict Risk", use_container_width=True)

        with col_result:
            if predict_clicked:

                # Build a single-row DataFrame in the same column order as training
                input_df  = pd.DataFrame([patient_input])[feature_names]
                risk_pct  = model.predict_proba(input_df)[0][1] * 100

                # ── Risk badge ─────────────────────────────────────────────
                if risk_pct >= CONFIG["high_risk_threshold"]:
                    bg, border, color, label = "#fef2f2","#fca5a5","#dc2626","🔴 HIGH RISK"
                elif risk_pct >= CONFIG["medium_risk_threshold"]:
                    bg, border, color, label = "#fffbeb","#fcd34d","#d97706","🟡 MEDIUM RISK"
                else:
                    bg, border, color, label = "#f0fdf4","#6ee7b7","#059669","🟢 LOW RISK"

                st.markdown(f"""
                <div style='background:{bg};border:1px solid {border};color:{color};
                            border-radius:8px;padding:14px;text-align:center;
                            font-size:1.3rem;font-weight:700;'>
                    {label} &nbsp;·&nbsp; {risk_pct:.1f}% No-Show Probability
                </div>""", unsafe_allow_html=True)

                # ── Progress bar ───────────────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                bar_col = (
                    COLORS["red"]   if risk_pct >= CONFIG["high_risk_threshold"] else
                    COLORS["amber"] if risk_pct >= CONFIG["medium_risk_threshold"] else
                    COLORS["green"]
                )
                fig, ax = plt.subplots(figsize=(5, 0.9))
                ax.barh([""], [100], color=COLORS["light"], height=0.5, edgecolor="none")
                ax.barh([""], [risk_pct], color=bar_col, height=0.5, edgecolor="none")
                ax.set_xlim(0, 100)
                ax.set_xlabel("No-Show Probability (%)", fontsize=9)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                ax.tick_params(left=False)
                fig.patch.set_facecolor("#ffffff")
                ax.set_facecolor("#ffffff")
                st.pyplot(fig); plt.close()

                # ── Intervention rules ─────────────────────────────────────
                # We pass the raw patient inputs + risk_pct to every rule.
                # The rules fire if their condition returns True.
                st.subheader("Recommended Actions")

                patient_ctx   = {**patient_input, "__risk_pct__": risk_pct}
                any_triggered = False

                for rule in INTERVENTION_RULES:
                    if rule["condition"](patient_ctx):
                        action_card(rule["message"])
                        any_triggered = True

                if not any_triggered:
                    action_card("✅ Low risk — a standard SMS reminder should be sufficient")

                # ── Revenue impact ─────────────────────────────────────────
                recovered = (risk_pct / 100) * cost_per_slot
                st.success(
                    f"💰 If the intervention works: "
                    f"~**${recovered:.0f}** in revenue protected for this slot"
                )

            else:
                st.markdown("""
                <div style='text-align:center;padding:60px 0;color:#9ca3af;'>
                    <div style='font-size:3rem;'>🏥</div>
                    <p style='margin-top:12px;'>
                        Adjust the inputs on the left<br>
                        and click <strong>Predict Risk</strong>
                    </p>
                </div>""", unsafe_allow_html=True)
