# Smart ML Prediction System

Streamlit app with four tools: house price, BMI, insurance cost, Titanic survival.
Models are trained on real public datasets (in `data/`) and scored on a held-out 20% test split.

| Model | Held-out score | Dumb baseline |
|---|---|---|
| House price (King County) | R² 0.51, MAE ≈ $159k | MAE ≈ $216k |
| Insurance charges | R² 0.88, MAE ≈ $2.4k | MAE ≈ $8.6k |
| Titanic survival | 78.8% accuracy | 61.5% |

## Run locally
    pip install -r requirements.txt
    python train_models.py     # optional, trained models are already in models/
    streamlit run app.py

## Deploy (Streamlit Community Cloud)
GitHub only stores the code. GitHub Pages cannot run Python and will show a 404.
1. Push this whole folder to a GitHub repo (keep `models/` and `data/`).
2. Go to share.streamlit.io, sign in with GitHub, click "Create app".
3. Pick the repo, branch `main`, main file `app.py`, then Deploy.
