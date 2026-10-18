# -*- coding: utf-8 -*-
"""
训练部署模型（全量 2008-2019，10 特征 XGBoost）+ 保存工件
输出：model/model.joblib, model/medians.json, model/metadata.json
"""
import os, json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, recall_score
from xgboost import XGBClassifier

BASE = r"C:\Users\Lixiao\Desktop\待升级-脓毒血症投稿"
OUT = os.path.join(BASE, "06_网页工具", "model")
os.makedirs(OUT, exist_ok=True)

F10 = ['crea','plt','ph','lact','resp','wbc','mbp','bun','rdw','bicar']
LABELS = {
    'crea': ("Creatinine", "mg/dL"),
    'plt': ("Platelet count", "×10⁹/L"),
    'ph': ("pH", ""),
    'lact': ("Lactate", "mmol/L"),
    'resp': ("Respiratory rate", "/min"),
    'wbc': ("White blood cell count", "×10⁹/L"),
    'mbp': ("Mean blood pressure", "mmHg"),
    'bun': ("Blood urea nitrogen", "mg/dL"),
    'rdw': ("Red cell distribution width", "%"),
    'bicar': ("Bicarbonate", "mmol/L"),
}

train = pd.read_csv(os.path.join(BASE, "04_重分析", "outputs", "cohort47_2008_2019.csv"))
X = train[F10].values.astype(float)
y = train['sepsis3'].astype(int).values

# 训练集中位数（插补用）
medians = {f: float(np.nanmedian(X[:, i])) for i, f in enumerate(F10)}
X_imp = np.where(np.isnan(X), np.array([medians[f] for f in F10]), X)

sp = (1 - y.mean()) / y.mean()
model = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, tree_method="hist", eval_metric="auc",
                      n_jobs=1, random_state=0, scale_pos_weight=sp).fit(X_imp, y)

# Youden 阈值（训练集内，作参考）
proba = model.predict_proba(X_imp)[:, 1]
best_j, best_thr = -1, 0.5
for thr in np.linspace(0.02, 0.98, 97):
    pred = (proba >= thr).astype(int)
    j = recall_score(y, pred) + recall_score(1-y, 1-pred) - 1
    if j > best_j:
        best_j, best_thr = j, thr

metadata = {
    "features": F10,
    "labels": {f: LABELS[f][0] for f in F10},
    "units": {f: LABELS[f][1] for f in F10},
    "medians": {f: round(medians[f], 3) for f in F10},
    "threshold": round(best_thr, 3),
    "training_n": int(len(y)),
    "training_prevalence": round(float(y.mean()), 3),
    "training_auc_in_sample": round(roc_auc_score(y, proba), 3),
    "model_type": "XGBoost (gradient-boosted trees)",
    "outcome": "Sepsis (Sepsis-3) within the suspected-infection window",
}

joblib.dump(model, os.path.join(OUT, "model.joblib"))
with open(os.path.join(OUT, "medians.json"), "w") as f:
    json.dump(medians, f, indent=2)
with open(os.path.join(OUT, "metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("=== 部署模型已保存 ===")
print(f"训练 n={len(y)}, 患病率 {y.mean()*100:.1f}%")
print(f"训练集内 AUC {roc_auc_score(y, proba):.3f}")
print(f"Youden 阈值 {best_thr:.3f}")
print(f"中位数: {medians}")
print(f"工件: {os.listdir(OUT)}")
