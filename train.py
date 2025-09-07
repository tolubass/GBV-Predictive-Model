import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score
import warnings

warnings.filterwarnings("ignore")


def load_and_prepare_data(data_path):
    """Load and prepare the GBV data"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}")

    df = pd.read_csv(data_path)

    target_col = "vulnerability_target"
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' column not found in data")

    print(f"Data loaded successfully. Shape: {df.shape}")
    print(f"Target distribution:\n{df[target_col].value_counts()}")

    return df, target_col


def train_logistic_regression_model(X_train, y_train):
    """Train Logistic Regression model with hyperparameter tuning"""

    # Create pipeline with scaling and logistic regression
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(solver="liblinear", max_iter=500, random_state=42))
    ])

    # Hyperparameters to tune
    param_grid = {
        "logreg__penalty": ["l1", "l2"],
        "logreg__C": [0.01, 0.1, 1, 10, 100],
        "logreg__class_weight": [None, "balanced"]
    }

    # Cross-validation strategy
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid Search
    print("🔄 Starting hyperparameter tuning...")
    grid_search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",
        cv=cv,
        verbose=1,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print(f"✅ Best parameters: {grid_search.best_params_}")
    print(f"✅ Best CV F1 Score: {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_, grid_search.best_params_


def extract_feature_importance(model, feature_names, top_n=10):
    """Extract feature importance from logistic regression coefficients"""

    # Get coefficients from the logistic regression step
    coefficients = model.named_steps["logreg"].coef_[0]

    # Create feature importance DataFrame
    feat_importance = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefficients,
        "Absolute_Importance": np.abs(coefficients)
    }).sort_values(by="Absolute_Importance", ascending=False)

    # Get top features
    top_features = feat_importance.head(top_n)["Feature"].tolist()

    print(f"✅ Top {top_n} most important features:")
    for i, (_, row) in enumerate(feat_importance.head(top_n).iterrows(), 1):
        print(f"  {i}. {row['Feature']}: {row['Absolute_Importance']:.4f}")

    return top_features, feat_importance


def train_final_model(X_train, y_train, top_features, best_params):
    """Train final model on top features using best parameters"""

    # Select top features
    X_train_top = X_train[top_features]

    # Create final pipeline with best parameters
    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            **{k.replace("logreg__", ""): v for k, v in best_params.items()},
            solver="liblinear",
            max_iter=500,
            random_state=42
        ))
    ])

    # Train final model
    final_model.fit(X_train_top, y_train)

    return final_model, X_train_top


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Evaluate model performance"""

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"\n✅ {model_name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  ROC AUC: {auc:.4f}")

    return {"accuracy": accuracy, "f1": f1, "auc": auc}


def save_model_artifacts(model, top_features, feature_importance, model_dir="model"):
    """Save model and related artifacts"""

    # Create model directory
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Save final model
    joblib.dump(model, f"{model_dir}/gbv_logistic_regression_model.joblib")
    print(f"✅ Model saved to {model_dir}/gbv_logistic_regression_model.joblib")

    # Save top features
    joblib.dump(top_features, f"{model_dir}/top_features.joblib")
    print(f"✅ Top features saved to {model_dir}/top_features.joblib")

    # Save feature importance
    feature_importance.to_csv(f"{model_dir}/feature_importance.csv", index=False)
    print(f"✅ Feature importance saved to {model_dir}/feature_importance.csv")

    # Save feature mappings for prediction (if needed for categorical features)
    # Note: Since you're using StandardScaler, the scaler is already part of the pipeline
    print(f"✅ All artifacts saved to {model_dir}/ directory")


def create_prediction_template(top_features, model_dir="model"):
    """Create a prediction template file"""

    template_data = {feature: 0 for feature in top_features}
    template_df = pd.DataFrame([template_data])
    template_df.to_csv(f"{model_dir}/prediction_template.csv", index=False)

    print(f"✅ Prediction template saved to {model_dir}/prediction_template.csv")
    print("   Use this template to understand required input format for predictions")


def main():
    """Main training function"""

    print("🚀 Starting GBV Vulnerability Prediction Model Training")
    print("=" * 60)

    # Load environment variables (optional)
    load_dotenv() if os.path.exists('.env') else None

    # Configuration
    DATA_PATH = os.getenv("DATA_PATH", r"C:\Users\hp\Desktop\gbv_projects\data\gendbv_data.csv")
    MODEL_DIR = os.getenv("MODEL_DIR", "model")
    TOP_N_FEATURES = int(os.getenv("TOP_N_FEATURES", "10"))

    try:
        # 1. Load and prepare data
        print("\n1️⃣ Loading and preparing data...")
        df, target_col = load_and_prepare_data(DATA_PATH)

        # Prepare features and target
        X = df.drop(columns=[target_col])
        y = df[target_col]

        # 2. Train/test split
        print("\n2️⃣ Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        print(f"✅ Training shape: {X_train.shape}")
        print(f"✅ Testing shape: {X_test.shape}")

        # 3. Train model with hyperparameter tuning
        print("\n3️⃣ Training model with hyperparameter tuning...")
        best_model, best_params = train_logistic_regression_model(X_train, y_train)

        # 4. Extract feature importance
        print(f"\n4️⃣ Extracting top {TOP_N_FEATURES} important features...")
        top_features, feature_importance = extract_feature_importance(
            best_model, X.columns, TOP_N_FEATURES
        )

        # 5. Train final model on top features
        print(f"\n5️⃣ Training final model on top {TOP_N_FEATURES} features...")
        final_model, X_train_top = train_final_model(X_train, y_train, top_features, best_params)

        # 6. Evaluate models
        print("\n6️⃣ Evaluating models...")

        # Full model evaluation
        full_metrics = evaluate_model(best_model, X_test, y_test, "Full Feature Model")

        # Top features model evaluation
        X_test_top = X_test[top_features]
        top_metrics = evaluate_model(final_model, X_test_top, y_test, "Top Features Model")

        # 7. Save model artifacts
        print("\n7️⃣ Saving model artifacts...")
        save_model_artifacts(final_model, top_features, feature_importance, MODEL_DIR)

        # 8. Create prediction template
        print("\n8️⃣ Creating prediction template...")
        create_prediction_template(top_features, MODEL_DIR)

        print("\n🎉 Model training completed successfully!")
        print("=" * 60)
        print(f"📁 Model saved in: {MODEL_DIR}/")
        print(f"🎯 Best F1 Score: {top_metrics['f1']:.4f}")
        print(f"🎯 Best Accuracy: {top_metrics['accuracy']:.4f}")
        print(f"🎯 Best ROC AUC: {top_metrics['auc']:.4f}")

    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        raise


if __name__ == "__main__":
    main()
