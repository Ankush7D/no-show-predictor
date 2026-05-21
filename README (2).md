# 🏥 Patient No-Show Predictor

> A machine learning web app that predicts patient no-shows and recommends real-time interventions for healthcare outreach agents — built end-to-end in Python.

🔗 **Live Demo:** [your-app.streamlit.app](https://your-app.streamlit.app) &nbsp;|&nbsp; ⭐ Star this repo if you find it useful!

---

## 🧠 The Problem I Solved

Healthcare providers lose **18–22% of appointment revenue** to no-shows. For a mid-sized clinic running 50 slots/day at $180 each, that's **~$1,600 lost every day** — over **$500,000 per year**.

I built a complete ML pipeline that:
- Predicts **which patients are likely to miss** their appointments
- Fires **rule-based intervention actions** (SMS, call, waitlist) tailored to each patient's risk profile
- Translates every prediction into an estimated **dollar value of protected revenue**

---

## 🖥️ Live App — 4 Pages

| Page | What it does |
|---|---|
| **📂 Data Setup** | Upload any CSV or load built-in demo data — no config needed |
| **📊 Data Explorer** | Visual EDA: gender, age, lead time, SMS impact, correlation heatmap |
| **🤖 Model Results** | ROC curve, confusion matrix, feature importances, 5-fold CV AUC |
| **🎯 Live Predictor** | Enter one patient's details → instant risk score + recommended actions |

---

## ⚙️ Technical Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit |
| **ML Model** | scikit-learn Random Forest (200 trees, balanced class weights) |
| **Data** | pandas, NumPy |
| **Visualisation** | Matplotlib, Seaborn |
| **Deployment** | Streamlit Community Cloud |

---

## 🔬 What I Built & How It Works

### 1. Auto-Preprocessing Pipeline
The app accepts **any CSV** — no hardcoded column names. It automatically:
- Drops columns with >50% missing values
- Label-encodes text columns (e.g. `"F"/"M"` → `0/1`)
- Fills nulls with column medians
- Selects only numeric features for training

### 2. Random Forest Classifier
```
n_estimators   = 200       # large ensemble for stability
max_depth      = 8         # prevents overfitting
class_weight   = balanced  # handles imbalanced show/no-show ratio
evaluation     = stratified 80/20 split + 5-fold cross-validation
typical AUC    = 0.72–0.76
```

### 3. Rule-Based Intervention Engine
A plug-and-play system where each rule is a Python dict with a `condition` lambda and an `action` message. Rules fire automatically based on patient data + risk score:

```python
{
    "condition": lambda p: p.get("sms_sent", 1) == 0,
    "message"  : "📱 No SMS sent — send a reminder (reduces no-shows by ~12%)",
},
{
    "condition": lambda p: p.get("prev_noshows", 0) >= 2,
    "message"  : "🔁 Repeat no-shower — consider double-booking or waitlist",
},
```
Adding a new rule = appending one dict. No code changes anywhere else.

### 4. Revenue Impact Calculator
```
Revenue protected = (Risk % / 100) × Revenue per slot ($180 default)
```
Every prediction shows the agent exactly how much money is on the line.

---

## 📊 Model Performance (Demo Data — 3,000 rows)

| Metric | Score |
|---|---|
| ROC-AUC | ~0.74 |
| 5-Fold CV AUC | ~0.72 ± 0.02 |
| Precision (No-Show class) | ~0.62 |
| Recall (No-Show class) | ~0.68 |

> AUC > 0.70 is considered clinically useful for appointment no-show prediction (see Kaggle Medical Appointment No-Shows benchmark).

---

## 🚀 Run Locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/no-show-predictor.git
cd no-show-predictor

# Install
pip install -r requirements.txt

# Launch
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

---

## 📂 Project Structure

```
no-show-predictor/
├── app.py            # Complete application — single file, heavily commented
├── requirements.txt
└── README.md
```

The entire ML pipeline, UI, intervention engine, and revenue calculator live in one well-structured file (~500 lines). Every section is documented with comments explaining the design decisions.

---

## 📋 Dataset Schema (Demo / Reference)

| Column | Type | Description |
|---|---|---|
| `age` | int | Patient age |
| `gender` | str | F / M |
| `lead_days` | int | Days between booking and appointment |
| `sms_sent` | 0/1 | Whether an SMS reminder was sent |
| `hypertension` | 0/1 | Patient has hypertension |
| `diabetes` | 0/1 | Patient has diabetes |
| `scholarship` | 0/1 | Patient receives welfare/scholarship |
| `appt_hour` | int | Hour of appointment (24h clock) |
| `prev_noshows` | int | Number of prior no-shows |
| `no_show` | 0/1 | **TARGET**: 1 = missed, 0 = showed up |

The demo data generator mirrors distributions from the real [Kaggle Medical Appointment No-Shows](https://www.kaggle.com/joniarroba/noshowappointments) dataset.

---

## 🔧 Configuration

All tunable parameters are in one place at the top of `app.py` — no hunting through code:

```python
CONFIG = {
    "cost_per_slot"       : 180,  # $ lost per missed slot
    "high_risk_threshold" : 60,   # % → RED badge
    "medium_risk_threshold": 35,  # % → AMBER badge
    "test_split"          : 0.20, # 20% held out for evaluation
    "n_trees"             : 200,
    "max_tree_depth"      : 8,
}
```

---

## 🗺️ What I'd Add Next

- [ ] REST API endpoint (`/score` POST) for direct integration with outreach pipelines
- [ ] SHAP values for per-patient explainability ("why is this patient high risk?")
- [ ] Batch scoring — upload tomorrow's appointments, export a ranked priority CSV


---

## 👤 About This Project

Built as part of a data science internship project to demonstrate how machine learning can create direct, measurable business impact in healthcare operations. The goal was to go beyond model accuracy and build something a real agent team could actually use.

If you're working on similar problems in healthcare ML, patient engagement, or outreach automation — I'd love to connect.

---

## 📄 License

MIT — free to use, fork, and adapt.
