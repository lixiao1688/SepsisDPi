# SepsisDPi — Early Sepsis Risk Prediction

A parsimonious 10-feature gradient-boosted model (XGBoost) for early sepsis risk prediction in ICU patients, developed and temporally validated on the MIMIC-IV database.

## Model

| | |
|---|---|
| Outcome | Sepsis (Sepsis-3) within the suspected-infection window |
| Features | 10 routine bedside/lab variables measured in the first 12 h of ICU admission |
| Model | XGBoost (400 trees, max_depth 6, learning_rate 0.05) |
| Training | MIMIC-IV 2008–2019 (n = 83,661 ICU stays) |
| Temporal external validation | MIMIC-IV 2020–2022 (AUC 0.820) |
| Internal validation | Patient-level 80/20, AUC 0.802 |

The 10 features are: creatinine, platelet count, pH, lactate, respiratory rate, white blood cell count, mean blood pressure, blood urea nitrogen, red cell distribution width, and bicarbonate. Missing values are imputed with the training-set median.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

The app is designed for [Streamlit Community Cloud](https://streamlit.io/cloud): connect this repository, select `app.py`, and deploy.

## Disclaimer

This tool is a research prototype for screening/decision support and is not a replacement for clinical judgement. It was developed on US ICU data (MIMIC-IV); performance in other populations has not been established. Reported discrimination is modest, and positive predictive value at the operating threshold implies roughly one false alert per true case.

## Data & reproducibility

The model was trained on the credentialed MIMIC-IV database (PhysioNet). Patient-level data are not shared. The training script (`train_deploy.py`) reproduces the model from the MIMIC-IV v3.1 data.
