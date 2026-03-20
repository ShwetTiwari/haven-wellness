import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score)
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_dataset
from feature_extractor import MultimodalFeatureExtractor

MODELS_DIR = Path(__file__).parent / "models" / "saved"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_ensemble():
    svm = SVC(kernel="rbf", C=10, gamma="scale", probability=True,
              class_weight="balanced", random_state=42)
    rf  = RandomForestClassifier(n_estimators=300, max_depth=10,
              class_weight="balanced", random_state=42, n_jobs=-1)
    gbc = GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
              max_depth=4, random_state=42)
    lr  = LogisticRegression(C=1.0, class_weight="balanced",
              max_iter=1000, random_state=42)
    return VotingClassifier(
        estimators=[("svm",svm),("rf",rf),("gbc",gbc),("lr",lr)],
        voting="soft", weights=[2,2,1,1]
    )


def print_metrics(y_true, y_pred, y_proba):
    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_true))>1 else float("nan")
    print(f"\n{'='*55}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  F1 Score : {f1:.4f}")
    print(f"  ROC AUC  : {auc:.4f}" if not np.isnan(auc) else "  ROC AUC  : N/A")
    print(classification_report(y_true, y_pred, labels=[0,1],
          target_names=["Not Depressed","Depressed"], zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")
    print(f"{'='*55}")
    return {"accuracy":float(acc),"f1":float(f1),"auc":0.0 if np.isnan(auc) else float(auc)}


def train():
    print("\n" + "="*55)
    print("  HAVEN — MULTIMODAL TRAINING (Option A: librosa WAV)")
    print("="*55)

    df = load_dataset()
    if len(df) < 4:
        raise ValueError(f"Only {len(df)} participants found.")

    # Check WAV files available
    wav_count = df["audio"].apply(lambda p: Path(str(p)).exists()).sum()
    print(f"\n[Data] {len(df)} participants | WAV files found: {wav_count}")
    if wav_count == 0:
        raise ValueError("No WAV files found! Check data directory.")

    test_size = 0.2 if len(df) >= 10 else 0.25
    df_train, df_test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=42
    )
    print(f"Train: {len(df_train)} | Test: {len(df_test)}")

    extractor = MultimodalFeatureExtractor()
    X_train   = extractor.fit_transform(df_train)
    X_test    = extractor.transform(df_test)
    y_train   = df_train["label"].values
    y_test    = df_test["label"].values

    print(f"\n[Training] Feature dims: {X_train.shape[1]}")
    print(f"           Depressed: {y_train.sum()} | Not: {(y_train==0).sum()}")
    print(f"           Audio features: {extractor.audio_ext.n_features} (librosa WAV)")

    print("\n-- Training ensemble (SVM + RF + GBC + LR) --")
    model = build_ensemble()
    model.fit(X_train, y_train)
    print("Training complete!")

    print("\n-- Test evaluation --")
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    metrics = print_metrics(y_test, y_pred, y_proba)

    n_folds = min(5, min(y_train.sum(), (y_train==0).sum()))
    if n_folds >= 2:
        print(f"\n-- {n_folds}-Fold Cross Validation --")
        cv = cross_val_score(
            build_ensemble(), X_train, y_train,
            cv=StratifiedKFold(n_folds, shuffle=True, random_state=42),
            scoring="accuracy", n_jobs=-1
        )
        print(f"  CV Accuracy: {cv.mean():.4f} +/- {cv.std():.4f}")
        metrics["cv_mean"] = float(cv.mean())
        metrics["cv_std"]  = float(cv.std())

    joblib.dump(model,     MODELS_DIR / "ensemble_model.pkl")
    joblib.dump(extractor, MODELS_DIR / "feature_extractor.pkl")
    pd.DataFrame([metrics]).to_csv(MODELS_DIR / "training_metrics.csv", index=False)

    print(f"\n-- Saved to {MODELS_DIR} --")
    print("  ensemble_model.pkl")
    print("  feature_extractor.pkl  (librosa WAV feature space)")
    print("  training_metrics.csv")
    print("\nDone! Run: python backend/app.py")
    return model, extractor, metrics


if __name__ == "__main__":
    train()