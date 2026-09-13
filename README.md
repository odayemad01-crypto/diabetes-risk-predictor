# Diabetes Risk Prediction — Streamlit App (based on oday_emadeldin_day_13.ipynb)

## What's included
- `app.py` — the Streamlit application (Prediction, Data Insights, Model Performance, About)
- `train_model.py` — reproduces the notebook's exact pipeline (drop duplicates → drop missing target → `pd.get_dummies(drop_first=True)` → 80/20 split → `StandardScaler` → `LogisticRegression`)
- `diabetes_model.joblib` — the trained model + scaler + feature column list
- `model_artifacts.json` — saved metrics, ROC curve, confusion matrix, feature importances
- `diabetes_sample.csv` — a cleaned data sample used for the EDA charts
- `requirements.txt` — Python dependencies

## Run it locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Retraining
```bash
python train_model.py
```
Regenerates the model file and artifacts — the app picks them up automatically.

## Deploying
Push this folder to a GitHub repo and deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud), pointing it at `app.py`.
