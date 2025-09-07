import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from sklearn.exceptions import NotFittedError
import warnings

warnings.filterwarnings("ignore")


class GBVVulnerabilityPredictor:
    """
    A class to handle GBV vulnerability predictions using the trained logistic regression model
    """

    def __init__(self, model_dir=r"C:\Users\hp\Desktop\gbv_projects\model"):
        """
        Initialize the predictor with model directory

        Args:
            model_dir (str): Directory containing the trained model and artifacts
        """
        self.model_dir = model_dir
        self.model = None
        self.top_features = None
        self.feature_importance = None
        self._is_loaded = False

    def load_model(self):
        """
        Load the trained model and associated artifacts
        """
        try:
            # Load the trained model - matches your train.py output filename
            model_path = os.path.join(self.model_dir, "gbv_logistic_regression_model.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            self.model = joblib.load(model_path)
            print(f"Model loaded successfully from {model_path}")

            # Load top features
            features_path = os.path.join(self.model_dir, "top_features.joblib")
            if not os.path.exists(features_path):
                raise FileNotFoundError(f"Features file not found: {features_path}")

            self.top_features = joblib.load(features_path)
            print(f"Top features loaded: {len(self.top_features)} features")

            # Load feature importance (optional)
            importance_path = os.path.join(self.model_dir, "feature_importance.csv")
            if os.path.exists(importance_path):
                self.feature_importance = pd.read_csv(importance_path)
                print("Feature importance loaded successfully")

            self._is_loaded = True
            return True

        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False

    def predict_single(self, input_data):
        """
        Make a prediction for a single individual

        Args:
            input_data (dict): Dictionary containing feature values

        Returns:
            dict: Prediction results including class, probability, and confidence
        """
        if not self._is_loaded:
            raise ValueError("Model not loaded. Please call load_model() first.")

        try:
            # Convert input to DataFrame
            if isinstance(input_data, dict):
                input_df = pd.DataFrame([input_data])
            else:
                input_df = pd.DataFrame(input_data)

            # Validate required features
            missing_features = set(self.top_features) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Select only the top features used in training
            input_df = input_df[self.top_features]

            # Make prediction
            prediction = self.model.predict(input_df)[0]
            probabilities = self.model.predict_proba(input_df)[0]
            confidence = round(probabilities[prediction] * 100, 2)

            # Prepare result
            result = {
                'prediction': int(prediction),
                'prediction_label': 'Vulnerable' if prediction == 1 else 'Not Vulnerable',
                'confidence': confidence,
                'probabilities': {
                    'not_vulnerable': round(probabilities[0] * 100, 2),
                    'vulnerable': round(probabilities[1] * 100, 2)
                },
                'risk_level': self._get_risk_level(probabilities[1])
            }

            return result

        except Exception as e:
            raise ValueError(f"Prediction error: {str(e)}")

    def predict_batch(self, input_data):
        """
        Make predictions for multiple individuals

        Args:
            input_data (pd.DataFrame or list of dicts): Input data for multiple predictions

        Returns:
            list: List of prediction results
        """
        if not self._is_loaded:
            raise ValueError("Model not loaded. Please call load_model() first.")

        try:
            # Convert input to DataFrame if needed
            if isinstance(input_data, list):
                input_df = pd.DataFrame(input_data)
            else:
                input_df = input_data.copy()

            # Validate required features
            missing_features = set(self.top_features) - set(input_df.columns)
            if missing_features:
                raise ValueError(f"Missing required features: {missing_features}")

            # Select only the top features used in training
            input_df = input_df[self.top_features]

            # Make predictions
            predictions = self.model.predict(input_df)
            probabilities = self.model.predict_proba(input_df)

            # Prepare results
            results = []
            for i in range(len(predictions)):
                pred = predictions[i]
                probs = probabilities[i]
                confidence = round(probs[pred] * 100, 2)

                result = {
                    'prediction': int(pred),
                    'prediction_label': 'Vulnerable' if pred == 1 else 'Not Vulnerable',
                    'confidence': confidence,
                    'probabilities': {
                        'not_vulnerable': round(probs[0] * 100, 2),
                        'vulnerable': round(probs[1] * 100, 2)
                    },
                    'risk_level': self._get_risk_level(probs[1])
                }
                results.append(result)

            return results

        except Exception as e:
            raise ValueError(f"Batch prediction error: {str(e)}")

    def _get_risk_level(self, vulnerability_probability):
        """
        Determine risk level based on vulnerability probability

        Args:
            vulnerability_probability (float): Probability of being vulnerable

        Returns:
            str: Risk level (Low, Medium, High, Very High)
        """
        if vulnerability_probability < 0.3:
            return "Low"
        elif vulnerability_probability < 0.5:
            return "Medium"
        elif vulnerability_probability < 0.75:
            return "High"
        else:
            return "Very High"

    def get_feature_importance(self, top_n=None):
        """
        Get feature importance rankings

        Args:
            top_n (int, optional): Number of top features to return

        Returns:
            pd.DataFrame: Feature importance rankings
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available")

        if top_n:
            return self.feature_importance.head(top_n)
        return self.feature_importance

    def get_required_features(self):
        """
        Get list of required features for prediction

        Returns:
            list: List of required feature names
        """
        if not self._is_loaded:
            raise ValueError("Model not loaded. Please call load_model() first.")
        return self.top_features.copy()

    def validate_input(self, input_data):
        """
        Validate input data format and features

        Args:
            input_data (dict or pd.DataFrame): Input data to validate

        Returns:
            dict: Validation results with status and messages
        """
        validation_result = {
            'is_valid': True,
            'messages': [],
            'missing_features': [],
            'extra_features': []
        }

        if not self._is_loaded:
            validation_result['is_valid'] = False
            validation_result['messages'].append("Model not loaded")
            return validation_result

        # Convert to DataFrame for validation
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        else:
            input_df = input_data

        # Check for missing features
        missing_features = set(self.top_features) - set(input_df.columns)
        if missing_features:
            validation_result['is_valid'] = False
            validation_result['missing_features'] = list(missing_features)
            validation_result['messages'].append(f"Missing required features: {missing_features}")

        # Check for extra features (informational only)
        extra_features = set(input_df.columns) - set(self.top_features)
        if extra_features:
            validation_result['extra_features'] = list(extra_features)
            validation_result['messages'].append(f"Extra features (will be ignored): {extra_features}")

        return validation_result


def load_model(model_dir=r"C:\Users\hp\Desktop\gbv_projects\model"):
    """
    Convenience function to load and return a GBV predictor instance

    Args:
        model_dir (str): Directory containing the trained model

    Returns:
        GBVVulnerabilityPredictor: Loaded predictor instance
    """
    try:
        predictor = GBVVulnerabilityPredictor(model_dir)
        success = predictor.load_model()

        if not success:
            raise ValueError("Failed to load model")

        return predictor
    except Exception as e:
        print(f"Error in load_model: {str(e)}")
        return None


def create_prediction_template(model_dir=r"C:\Users\hp\Desktop\gbv_projects\model"):
    """
    Create a template showing required input format

    Args:
        model_dir (str): Directory containing model artifacts

    Returns:
        dict: Template dictionary with feature names and example values
    """
    try:
        features_path = os.path.join(model_dir, "top_features.joblib")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        top_features = joblib.load(features_path)

        # Create template with placeholder values
        template = {feature: 0 for feature in top_features}

        return template

    except Exception as e:
        raise ValueError(f"Error creating template: {str(e)}")


def get_model_info(model_dir=r"C:\Users\hp\Desktop\gbv_projects\model"):
    """
    Get information about the trained model

    Args:
        model_dir (str): Directory containing model artifacts

    Returns:
        dict: Model information including features and performance
    """
    info = {
        'model_available': False,
        'features_count': 0,
        'required_features': [],
        'feature_importance_available': False
    }

    try:
        # Check if model exists - matches your train.py output filename
        model_path = os.path.join(model_dir, "gbv_logistic_regression_model.joblib")
        info['model_available'] = os.path.exists(model_path)

        # Check features
        features_path = os.path.join(model_dir, "top_features.joblib")
        if os.path.exists(features_path):
            features = joblib.load(features_path)
            info['features_count'] = len(features)
            info['required_features'] = features

        # Check feature importance
        importance_path = os.path.join(model_dir, "feature_importance.csv")
        info['feature_importance_available'] = os.path.exists(importance_path)

        return info

    except Exception as e:
        print(f"Error getting model info: {str(e)}")
        return info


def example_prediction():
    """
    Example of how to use the predictor
    """
    try:
        # Load the model
        predictor = load_model()
        if predictor is None:
            print("Failed to load predictor")
            return None

        # Example input data (replace with actual feature values)
        sample_input = create_prediction_template()

        # Update with some example values
        sample_input.update({
            'individual_age': 25,
            'gender': 0,  # 0 = Female, 1 = Male
            'individual_employment_status': 1,  # 1 = Employed
            'current_living_arrangement': 1,  # Not alone
            'educational_status': 3,  # Some secondary
        })

        # Make prediction
        result = predictor.predict_single(sample_input)

        print("Prediction Result:")
        print(f"Prediction: {result['prediction_label']}")
        print(f"Confidence: {result['confidence']}%")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Probabilities: {result['probabilities']}")

        return result

    except Exception as e:
        print(f"Example prediction failed: {str(e)}")
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test the model utilities
    print("Testing GBV Model Utilities")
    print("=" * 40)

    # Get model info
    model_info = get_model_info()
    print("Model Info:")
    for key, value in model_info.items():
        print(f"  {key}: {value}")

    # Run example prediction if model is available
    if model_info['model_available']:
        print("\nRunning example prediction...")
        example_prediction()
    else:
        print("\nModel not available. Please run train.py first.")