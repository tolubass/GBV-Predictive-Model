import pandas as pd
import numpy as np
import shap
import json
import logging
from model_utils import GBVVulnerabilityPredictor
import warnings

warnings.filterwarnings("ignore")

# -----------------------------
# Rule-based insights
# -----------------------------
def gbv_rule_insights(instance_data):
    """Simple domain rules for GBV vulnerability insights."""
    reasons, suggestions = [], []

    age = instance_data.get("individual_age")
    if age is not None:
        if age < 18:
            reasons.append("Individual is under 18, potentially more vulnerable to GBV.")
            suggestions.append("Consider additional protective interventions for minors.")
        elif age > 50:
            reasons.append("Individual is above 50, vulnerability risk may vary.")
            suggestions.append("Monitor and provide supportive community resources.")

    employment = instance_data.get("individual_employment_status")
    if employment == 0:  # Assuming 0 = unemployed
        reasons.append("Unemployment may increase vulnerability.")
        suggestions.append("Provide economic support or counseling services.")

    education = instance_data.get("educational_status")
    if education is not None and education <= 2:
        reasons.append("Low education level may affect awareness and reporting.")
        suggestions.append("Consider targeted awareness campaigns.")

    return reasons, suggestions

# -----------------------------
# SHAP explainer caching
# -----------------------------
_shap_explainer_cache = {}

def get_shap_explainer(model):
    """Return cached SHAP LinearExplainer for the model."""
    model_id = id(model)
    if model_id not in _shap_explainer_cache:
        try:
            _shap_explainer_cache[model_id] = shap.LinearExplainer(
                model.named_steps["logreg"],
                model.named_steps["scaler"].transform(
                    np.zeros((1, len(model.named_steps["logreg"].coef_[0])))
                ),
                feature_perturbation="interventional"
            )
        except Exception as e:
            logging.error(f"Failed to create SHAP explainer: {e}")
            _shap_explainer_cache[model_id] = None
    return _shap_explainer_cache[model_id]

# -----------------------------
# Explanation functions
# -----------------------------
def explain_instance(predictor: GBVVulnerabilityPredictor, input_dict: dict, top_n=5):
    """Generate prediction and explanation for a single instance."""
    if not predictor._is_loaded:
        raise ValueError("Predictor not loaded. Call load_model() first.")

    input_df = pd.DataFrame([input_dict])
    missing_features = set(predictor.top_features) - set(input_df.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    input_df = input_df[predictor.top_features]

    # Prediction
    pred_result = predictor.predict_single(input_dict)
    prediction_label = pred_result["prediction_label"]
    confidence = pred_result["confidence"]

    # SHAP values
    explainer = get_shap_explainer(predictor.model)
    shap_values = explainer.shap_values(input_df) if explainer else None

    if shap_values is not None:
        if isinstance(shap_values, list):
            shap_array = shap_values[1][0]
        else:
            shap_array = shap_values[0]
        shap_dict = dict(zip(input_df.columns, shap_array))
        top_features = {k: round(float(v), 4) for k, v in sorted(shap_dict.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]}
    else:
        top_features = {}

    reasons, suggestions = gbv_rule_insights(input_dict)

    explanation = {
        "prediction": {"label": prediction_label, "confidence": round(confidence, 2)},
        "top_features": top_features,
        "insights": {"reasons": reasons, "suggestions": suggestions}
    }

    return explanation

def explain_batch(predictor: GBVVulnerabilityPredictor, input_list: list, top_n=5):
    """Generate explanations for a batch of instances, returned as JSON."""
    results = []
    for idx, instance in enumerate(input_list, start=1):
        try:
            explanation = explain_instance(predictor, instance, top_n=top_n)
            results.append({"instance": idx, "explanation": explanation})
        except Exception as e:
            logging.error(f"Error explaining instance {idx}: {e}")
            results.append({"instance": idx, "error": str(e)})
    return json.dumps(results, indent=4)

def explain_instance_json(predictor: GBVVulnerabilityPredictor, input_dict: dict, top_n=5):
    """Return single instance explanation as JSON string."""
    explanation = explain_instance(predictor, input_dict, top_n=top_n)
    return json.dumps(explanation, indent=4)

# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    predictor = GBVVulnerabilityPredictor()
    predictor.load_model()

    # Single instance
    sample_input = {feat: 0 for feat in predictor.top_features}
    sample_input.update({
        "individual_age": 25,
        "gender": 0,
        "individual_employment_status": 1,
        "current_living_arrangement": 1,
        "educational_status": 3
    })

    print("\n=== Single GBV Vulnerability Explanation (JSON) ===")
    print(explain_instance_json(predictor, sample_input))

    # Batch instances
    batch_inputs = [
        sample_input,
        {**sample_input, "individual_age": 16, "individual_employment_status": 0, "educational_status": 2}
    ]

    print("\n=== Batch GBV Vulnerability Explanations (JSON) ===")
    print(explain_batch(predictor, batch_inputs))

