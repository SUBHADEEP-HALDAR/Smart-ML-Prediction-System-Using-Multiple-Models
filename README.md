# Smart ML Prediction System

A Streamlit web app with four tools: house price, BMI, insurance cost and Titanic survival.

**Live app:** https://smart-ml-predictor-sh.streamlit.app
(Free hosting sleeps when unused. If you see a "gone to sleep" screen, click the wake-up button and wait about 30 seconds.)

## The four tools

| Tool | What it does | Data |
|---|---|---|
| House Price | Estimates price from living area, bedrooms and bathrooms | King County, WA (USA), 21k home sales |
| BMI | Weight / height² with a health category (a formula, not machine learning) | none |
| Insurance Cost | Estimates yearly medical insurance cost from age, sex, BMI, children, smoker, region | 1,338 people (USA) |
| Titanic Survival | Predicts survival from class, sex and age, with a probability | 891 real passengers |

## Model quality

Each model was scored on a 20% test split it never saw during training, and compared with a dumb baseline.

| Model | Method | Test score | Dumb baseline |
|---|---|---|---|
| House price | Gradient boosting | R² 0.51, typical error about $159k | error about $216k |
| Insurance | Gradient boosting | R² 0.88, typical error about $2.4k | error about $8.6k |
| Titanic | Random forest | 78.8% accuracy | 61.5% (always guess "did not survive") |

## Known limitations

- **The house model is weak.** Area, bedrooms and bathrooms explain only about half of price. Location drives the rest, and the model doesn't have it.
- **The data is American.** Prices are in US dollars and are converted to rupees in the app using an exchange rate you can change in the sidebar (default 96 ₹ per $). Treat rupee figures as a rough guide. They do not reflect Indian house or insurance prices.
- **Titanic:** the model sees only class, sex and age. It is a demonstration, not a serious predictor.
- BMI ignores muscle mass, age and body composition.

## Project files

    app.py             the Streamlit app
    train_models.py    trains the three models and saves them
    requirements.txt   Python packages
    *.csv              training data (house, insurance, titanic)
    *_model.joblib     trained models
    metrics.json       test scores shown in the app

## Run locally

    pip install -r requirements.txt
    streamlit run app.py

To retrain the models: `python train_models.py`

## Tech stack

Python, Streamlit, scikit-learn, pandas, joblib. Hosted on Streamlit Community Cloud.
