import argparse
import sys
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def find_dataset(provided_path=None):
    candidates = []
    if provided_path:
        candidates.append(Path(provided_path))
    candidates += [
        Path("dataset.csv"),
        Path("data/dataset.csv"),
        Path("mental_health_data.csv"),
        Path("data/mental_health_data.csv"),
        Path("dataset (1).csv")
    ]
    for p in candidates:
        if p and p.exists():
            return p.resolve()
    return None

def main():
    parser = argparse.ArgumentParser(description="Train improved model for mental health prediction")
    parser.add_argument("--data", "-d", help="Path to CSV dataset (optional)")
    args = parser.parse_args()

    dataset_path = find_dataset(args.data)
    if not dataset_path:
        print("❌ Dataset not found.")
        sys.exit(1)

    print(f"✅ Using dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
    df.columns = df.columns.str.strip().str.lower()

    if "treatment" not in df.columns:
        print("❌ 'treatment' column missing.")
        sys.exit(1)

    df.drop(['timestamp', 'comments', 'state', 'country'], axis=1, inplace=True, errors='ignore')
    df = df.dropna(how='all')
    df = df.ffill().bfill()

    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop("treatment", axis=1)
    y = df["treatment"]

    if y.dtype == 'O' or str(y.dtype) == 'category':
        le_y = LabelEncoder()
        y = le_y.fit_transform(y.astype(str))
        label_encoders['_target_treatment'] = le_y

    # Balance dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(f"✅ Data balanced using SMOTE: {len(X_resampled)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # XGBoost with randomized search
    model = XGBClassifier(random_state=42, eval_metric='mlogloss')

    param_dist = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
    }

    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=20,
        cv=4,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_

    print(f"\n✅ Best Parameters: {random_search.best_params_}")

    # Evaluate model
    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc * 100:.2f}%\n")
    print("📊 Classification Report:\n", classification_report(y_test, y_pred))

    # Cross validation score
    cv_scores = cross_val_score(best_model, X_resampled, y_resampled, cv=5, scoring='accuracy')
    print(f"✅ Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")

    # Feature importance plot
    importance = best_model.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(sorted_idx)), importance[sorted_idx], align="center")
    plt.xticks(range(len(sorted_idx)), np.array(X.columns)[sorted_idx], rotation=90)
    plt.title("Feature Importance (XGBoost)")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    print("📈 Saved feature importance plot -> feature_importance.png")

    # Save model and encoders
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    joblib.dump(best_model, out_dir / "model.pkl")
    joblib.dump(label_encoders, out_dir / "encoders.pkl")

    print(f"✅ Saved model -> {out_dir / 'model.pkl'}")
    print(f"✅ Saved encoders -> {out_dir / 'encoders.pkl'}")
    print("🎯 Training complete with enhanced accuracy ✅")

if __name__ == "__main__":
    main()
