"""Smart ML Prediction System - Streamlit app.

Run locally:  streamlit run app.py
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent


def find(name):
    """Find a file in models/ or next to app.py (works with a flat GitHub upload too)."""
    for folder in (ROOT / "models", ROOT):
        if (folder / name).exists():
            return folder / name
    raise FileNotFoundError(name)

st.set_page_config(page_title="Smart ML Prediction System", page_icon="🤖", layout="centered")


USD_TO_INR = 96.0  # approximate rate (about 96.1 on 2 Oct 2026); adjustable in the sidebar


def inr(usd):
    """Convert a US-dollar amount to rupees, formatted with Indian digit grouping (12,34,567)."""
    n = int(round(usd * rate))
    s = str(abs(n))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return ("-" if n < 0 else "") + "₹" + s


@st.cache_resource
def load_assets():
    """Load models + metrics once. Fail loudly instead of silently faking results."""
    names = ["house", "insurance", "titanic"]
    models = {n: joblib.load(find(f"{n}_model.joblib")) for n in names}
    metrics = json.loads(find("metrics.json").read_text())
    return models, metrics


try:
    MODELS, METRICS = load_assets()
except Exception as exc:  # missing/incompatible model files
    st.error(
        "Models could not be loaded. Run `python train_models.py` to rebuild them.\n\n"
        f"Details: `{exc}`"
    )
    st.stop()


def range_warning(label, value, rng):
    lo, hi = rng
    if value < lo or value > hi:
        st.warning(
            f"{label} = {value:g} is outside the range the model was trained on "
            f"({lo:g} to {hi:g}). Treat this prediction as unreliable."
        )


st.title("🤖 Smart ML Prediction System")
st.caption("Four small tools. Models are trained on real public datasets and evaluated on held-out data.")

option = st.sidebar.selectbox(
    "Select tool", ("House Price", "BMI", "Insurance Cost", "Titanic Survival")
)

rate = st.sidebar.number_input(
    "USD → INR rate", min_value=50.0, max_value=150.0, value=USD_TO_INR, step=0.5,
    help="The datasets are in US dollars. Prices are converted to rupees with this rate.",
)

# ───────────────────────── HOUSE ─────────────────────────
if option == "House Price":
    m = METRICS["house"]
    st.header("🏠 House Price Prediction")
    st.caption("King County, WA (USA) sales, 21k homes. US prices converted to rupees, so treat them as a rough guide only.")

    area = st.number_input("Living area (sq ft)", min_value=200, max_value=15000, value=1800, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=0.5, max_value=8.0, value=2.0, step=0.25)

    if st.button("Predict price", type="primary"):
        range_warning("Area", area, m["ranges"]["sqft_living"])
        X = pd.DataFrame([{"sqft_living": area, "bedrooms": bedrooms, "bathrooms": bathrooms}])
        price = float(MODELS["house"].predict(X)[0])
        st.success(f"Estimated price: {inr(price)}")
        st.info(
            f"Typical error on unseen homes is about ±{inr(m['mae'])} "
            f"(R² = {m['r2']}). Size, bedrooms and bathrooms explain only about half of price: "
            "location is the biggest driver and this model does not know it."
        )

# ───────────────────────── BMI ─────────────────────────
elif option == "BMI":
    st.header("🧍 BMI Calculator")
    st.caption("This is a formula, not machine learning: weight / height².")

    height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.70, step=0.01)
    weight = st.number_input("Weight (kg)", min_value=10.0, max_value=400.0, value=70.0, step=0.5)

    if st.button("Calculate BMI", type="primary"):
        bmi = weight / height**2
        st.metric("BMI", f"{bmi:.1f}")
        if bmi < 18.5:
            st.warning("Underweight")
        elif bmi < 25:
            st.success("Normal")
        elif bmi < 30:
            st.warning("Overweight")
        else:
            st.error("Obese range")
        st.caption("BMI is a rough screening number. It ignores muscle mass, age and body composition.")

# ───────────────────────── INSURANCE ─────────────────────────
elif option == "Insurance Cost":
    m = METRICS["insurance"]
    st.header("💰 Insurance Cost Prediction")
    st.caption("Medical insurance dataset (1,338 people, USA). Yearly cost, US dollars converted to rupees.")

    c1, c2 = st.columns(2)
    age = c1.number_input("Age", min_value=18, max_value=100, value=35)
    sex = c2.selectbox("Sex", ("female", "male"))
    bmi = c1.number_input("BMI", min_value=10.0, max_value=60.0, value=26.0, step=0.1)
    children = c2.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = c1.selectbox("Smoker", ("no", "yes"))
    region = c2.selectbox("Region", ("northeast", "northwest", "southeast", "southwest"))

    if st.button("Predict yearly cost", type="primary"):
        range_warning("Age", age, m["ranges"]["age"])
        range_warning("BMI", bmi, m["ranges"]["bmi"])
        X = pd.DataFrame([{"age": age, "sex": sex, "bmi": bmi, "children": children,
                           "smoker": smoker, "region": region}])
        cost = float(MODELS["insurance"].predict(X)[0])
        st.success(f"Estimated yearly cost: {inr(cost)}")
        st.info(f"Typical error on unseen people is about ±{inr(m['mae'])} (R² = {m['r2']}).")

# ───────────────────────── TITANIC ─────────────────────────
else:
    m = METRICS["titanic"]
    st.header("🚢 Titanic Survival Prediction")
    st.caption("Trained on 891 real passengers.")

    pclass = st.selectbox("Passenger class", (1, 2, 3), index=2)
    sex = st.selectbox("Sex", ("female", "male"))
    age = st.number_input("Age", min_value=0, max_value=100, value=30)

    if st.button("Predict survival", type="primary"):
        X = pd.DataFrame([{"pclass": pclass, "sex": sex, "age": age}])
        model = MODELS["titanic"]
        p = float(model.predict_proba(X)[0][1])
        if p >= 0.5:
            st.success(f"Likely survived ({p:.0%} survival probability)")
        else:
            st.error(f"Likely did not survive ({p:.0%} survival probability)")
        st.progress(p)
        st.info(
            f"Accuracy on unseen passengers: {m['accuracy']:.0%}, versus {m['baseline_accuracy']:.0%} "
            "for always guessing 'did not survive'."
        )

st.markdown("---")
with st.expander("Model quality (held-out test data)"):
    h, i, t = METRICS["house"], METRICS["insurance"], METRICS["titanic"]
    st.table(pd.DataFrame(
        [
            ["House price", f"R² {h['r2']}", f"MAE {inr(h['mae'])} vs {inr(h['baseline_mae'])} for guessing the median"],
            ["Insurance", f"R² {i['r2']}", f"MAE {inr(i['mae'])} vs {inr(i['baseline_mae'])} for guessing the median"],
            ["Titanic", f"Accuracy {t['accuracy']:.1%}", f"{t['baseline_accuracy']:.1%} for always guessing 'died'"],
        ],
        columns=["Model", "Score", "Compared with a dumb baseline"],
    ).set_index("Model"))
