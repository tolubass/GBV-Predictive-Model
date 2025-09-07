
from flask import Flask, request, jsonify, render_template_string
import numpy as np
import base64
import io
import matplotlib

# Use non-interactive backend for matplotlib (works on servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_utils import GBVVulnerabilityPredictor
from Explanation_engine import explain_instance

import warnings
import logging
import threading
import webbrowser
import time

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.secret_key = "gbv-prediction-app-2024"

# Initialize predictor globally (single instance)
predictor = None


def initialize_predictor():
    """Initialize the GBV predictor on app startup"""
    global predictor
    try:
        predictor = GBVVulnerabilityPredictor()
        success = predictor.load_model()
        if not success:
            raise Exception("Failed to load model")
        logging.info("Model loaded successfully")
        return True
    except Exception as e:
        logging.error(f"Failed to initialize predictor: {e}")
        predictor = None
        return False


# Minimal HTML template (keeps styling but only one template string).
# NOTE: This template is embedded inside the Python file to make this a single-file app.
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>GBV Vulnerability Assessment</title>
    <style>
      body { font-family: Arial, Helvetica, sans-serif; margin: 20px; background: #f5f7fb; }
      .card { background: white; padding: 18px; border-radius: 8px; box-shadow: 0 6px 18px rgba(0,0,0,0.06); max-width: 900px; margin: auto; }
      label { display:block; margin-top:10px; font-weight:600; }
      input, select { width:100%; padding:8px; margin-top:6px; border-radius:6px; border:1px solid #ddd; }
      .row { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
      .full { grid-column: 1 / -1; }
      button { margin-top:14px; padding:10px 16px; border:none; background:#4f46e5; color:white; border-radius:8px; cursor:pointer; }
      .results { margin-top: 18px; }
      .badge { display:inline-block; padding:6px 10px; border-radius:6px; color:white; font-weight:700; }
      .vulnerable { background: linear-gradient(90deg,#ef4444,#f97316); }
      .not-vulnerable { background: linear-gradient(90deg,#10b981,#34d399); }
      .section-title { margin-top:14px; font-size:1.05rem; font-weight:700; }
      ul { padding-left: 18px; }
      img.shap { max-width:100%; height:auto; border-radius:6px; margin-top:8px; }
      .small { font-size:0.9rem; color:#666; }
      @media (max-width:700px){ .row { grid-template-columns: 1fr } }
    </style>
  </head>
  <body>
    <div class="card">
      <h2>GBV Vulnerability Assessment</h2>
      <p class="small">Fill required fields and click "Assess Vulnerability Risk".</p>

      <form id="predictionForm">
        <div class="row">
          <div>
            <label for="individual_age">Age</label>
            <input type="number" id="individual_age" name="individual_age" min="1" max="100" required>
          </div>
          <div>
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
              <option value="">Select...</option>
              <option value="0">Female</option>
              <option value="1">Male</option>
            </select>
          </div>

          <div>
            <label for="current_living_arrangement">Living Situation</label>
            <select id="current_living_arrangement" name="current_living_arrangement" required>
              <option value="">Select...</option>
              <option value="0">Alone</option>
              <option value="1">Parent</option>
              <option value="2">Guardian</option>
              <option value="3">Partner</option>
              <option value="-1">Unknown</option>
              <option value="99">Other</option>
            </select>
          </div>

          <div>
            <label for="individual_lives_with">Household Composition</label>
            <select id="individual_lives_with" name="individual_lives_with" required>
              <option value="">Select...</option>
              <option value="0">Alone</option>
              <option value="1">Parent</option>
              <option value="2">Parent/Guardian</option>
              <option value="3">Relative</option>
              <option value="4">Spouse/Cohabiting</option>
              <option value="-1">Unknown</option>
            </select>
          </div>

          <div>
            <label for="individual_employment_status">Employment status</label>
            <select id="individual_employment_status" name="individual_employment_status" required>
              <option value="">Select...</option>
              <option value="0">Unemployed</option>
              <option value="1">Currently employed</option>
              <option value="2">Self employed</option>
              <option value="3">Not reported</option>
              <option value="-1">Unknown</option>
              <option value="99">Other</option>
            </select>
          </div>

          <div>
            <label for="educational_status">Education level</label>
            <select id="educational_status" name="educational_status" required>
              <option value="">Select...</option>
              <option value="0">No formal education</option>
              <option value="1">Some primary</option>
              <option value="2">Completed primary</option>
              <option value="3">Some secondary</option>
              <option value="4">Completed secondary</option>
              <option value="5">Undergraduate</option>
              <option value="6">Graduate</option>
              <option value="7">Postgraduate</option>
              <option value="8">Diploma</option>
              <option value="-1">Unknown</option>
            </select>
          </div>

          <div>
            <label for="househead_employment_status">Head of Household Employment</label>
            <select id="househead_employment_status" name="househead_employment_status" required>
              <option value="">Select...</option>
              <option value="0">Unemployed</option>
              <option value="1">Currently employed</option>
              <option value="2">Self employed</option>
              <option value="3">Not reported</option>
              <option value="-1">Unknown</option>
              <option value="99">Other</option>
            </select>
          </div>

          <div>
            <label for="vul_cat_if_applicable">Previously victimized?</label>
            <select id="vul_cat_if_applicable" name="vul_cat_if_applicable" required>
              <option value="">Select...</option>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>

          <div>
            <label for="IDP">Are you an internally displaced person(IDP)?</label>
            <select id="IDP" name="IDP" required>
              <option value="">Select...</option>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>

          <div>
            <label for="PLWD">Any disability?</label>
            <select id="PLWD" name="PLWD" required>
              <option value="">Select...</option>
              <option value="0">No</option>
              <option value="1">Yes</option>
            </select>
          </div>

        </div>

        <div class="full">
          <button type="submit" id="predictBtn">Assess Vulnerability Risk</button>
        </div>
      </form>

      <div class="results" id="results" style="display:none;">
        <div>
          <h3>Result</h3>
          <div id="predictionBadge"></div>
          <p id="predictionLabel"></p>
          <p id="confidenceScore"></p>
          <p id="riskLevel"></p>
        </div>

        <div>
          <h4 class="section-title">AI Summary</h4>
          <div id="summaryInsights"></div>
        </div>

        <div>
          <h4 class="section-title">Top Contributing Factors (SHAP)</h4>
          <div id="shapPlot"></div>
        </div>

        <div>
          <h4 class="section-title">Model Input</h4>
          <div id="inputDisplay"></div>
        </div>
      </div>
    </div>

    <script>
      // Submit form via fetch to /predict
      document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        document.getElementById('predictBtn').disabled = true;

        const form = new FormData(this);
        const data = {};
        for (let [k,v] of form.entries()) data[k]=v;

        try {
          const res = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type':'application/json'},
            body: JSON.stringify(data)
          });
          const payload = await res.json();
          if (!payload.success) {
            alert(payload.error || 'Prediction failed');
            return;
          }
          renderResult(payload);
        } catch(err) {
          alert('Network or server error: ' + err.message);
        } finally {
          document.getElementById('predictBtn').disabled = false;
        }
      });

      function renderResult(payload) {
        const p = payload.prediction || {};
        const explanation = payload.explanation || {};
        const user_input = payload.user_input || payload.model_input || {};

        // Badge
        const isVul = (p.prediction === 1 || p.prediction === '1' || p.prediction === 'VULNERABLE');
        const badge = document.getElementById('predictionBadge');
        badge.innerHTML = isVul ? '<span class="badge vulnerable">VULNERABLE</span>' : '<span class="badge not-vulnerable">NOT VULNERABLE</span>';

        document.getElementById('predictionLabel').textContent = p.prediction_label || (isVul ? 'VULNERABLE' : 'NOT VULNERABLE');
        document.getElementById('confidenceScore').textContent = 'Confidence: ' + (p.confidence !== undefined ? p.confidence + '%' : (p.prob && (p.prob*100).toFixed(1)+'%') || 'N/A');
        document.getElementById('riskLevel').textContent = 'Risk Level: ' + (p.risk_level || 'N/A');

        // Insights
        let s = '';
        if (explanation && explanation.insights) {
          if (explanation.insights.reasons && explanation.insights.reasons.length) {
            s += '<h5>Key Risk Factors</h5><ul>';
            explanation.insights.reasons.forEach(r => s += `<li>${r}</li>`);
            s += '</ul>';
          }
          if (explanation.insights.suggestions && explanation.insights.suggestions.length) {
            s += '<h5>Recommended Interventions</h5><ul>';
            explanation.insights.suggestions.forEach(r => s += `<li>${r}</li>`);
            s += '</ul>';
          }
        } else {
          s += '<p class="small">No insights available.</p>';
        }
        document.getElementById('summaryInsights').innerHTML = s;

        // SHAP
        if (payload.shap_plot) {
          document.getElementById('shapPlot').innerHTML = `<img class="shap" src="data:image/png;base64,${payload.shap_plot}" alt="shap">`;
        } else {
          document.getElementById('shapPlot').innerHTML = '<p class="small">No SHAP plot available.</p>';
        }

        // Input display
        let inputHTML = '<div>';
        for (const k of Object.keys(user_input)) {
          inputHTML += `<div><strong>${k}</strong>: ${user_input[k]}</div>`;
        }
        inputHTML += '</div>';
        document.getElementById('inputDisplay').innerHTML = inputHTML;

        document.getElementById('results').style.display = 'block';
        window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' });
      }
    </script>
  </body>
</html>
"""


def safe_int_cast(v, default=0):
    """Try to cast to int, return default if fail."""
    try:
        return int(v)
    except Exception:
        try:
            return int(float(v))
        except Exception:
            return default


def synthesize_insights(top_features, top_n=5):
    """
    If explain_instance returns empty insights, create fallback reasons and suggestions
    using the sign of the feature impacts.
    """
    reasons = []
    suggestions = []
    if not top_features:
        return {"reasons": reasons, "suggestions": suggestions}

    # sort by absolute impact descending
    items = sorted(top_features.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    for feat, impact in items:
        direction = "increases" if impact > 0 else "decreases"
        reasons.append(f"{feat} {direction} vulnerability (impact: {impact:.3f})")
        # basic suggestions per feature
        if impact > 0:
            suggestions.append(f"Monitor {feat} closely and consider targeted support/intervention for factors related to {feat}.")
        else:
            suggestions.append(f"{feat} appears protective; promote/maintain factors associated with {feat} where appropriate.")
    return {"reasons": reasons, "suggestions": suggestions}


@app.route("/")
def index():
    if predictor is None:
        return "Model not loaded. Check server logs.", 500
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    global predictor
    if predictor is None:
        return jsonify({"success": False, "error": "Model not loaded."})

    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"success": False, "error": "Invalid input format; expected JSON object."})

        # Convert to model_input dict (int-cast values)
        model_input = {}
        for k, v in data.items():
            model_input[k] = safe_int_cast(v, default=0)

        # Ensure required features are present
        try:
            required = predictor.get_required_features()
            if isinstance(required, (list, tuple)):
                for feat in required:
                    if feat not in model_input:
                        model_input[feat] = 0
        except Exception:
            # If predictor doesn't expose get_required_features, ignore
            pass

        # Run prediction
        raw_pred = predictor.predict_single(model_input)

        # Normalize prediction structure
        if isinstance(raw_pred, dict):
            prediction = raw_pred
        else:
            # if predictor returns an int or label, craft a stable structure
            pred_int = int(raw_pred) if raw_pred is not None else 0
            prediction = {
                "prediction": pred_int,
                "prediction_label": "VULNERABLE" if pred_int == 1 else "NOT VULNERABLE",
                "confidence": 100.0 if pred_int in (0, 1) else 0.0,
                "risk_level": "High" if pred_int == 1 else "Low"
            }

        # Get explanation (shap + insights) - wrap in try
        try:
            explanation = explain_instance(predictor, model_input, top_n=5)
            if explanation is None:
                explanation = {}
        except Exception as e:
            logging.exception("explain_instance failed: %s", e)
            explanation = {}

        # Ensure explanation has top_features and insights keys
        top_features = explanation.get("top_features") or {}
        insights = explanation.get("insights") or {"reasons": [], "suggestions": []}

        # If no insights, synthesize fallback ones from top_features
        if not insights.get("reasons") and not insights.get("suggestions"):
            synthesized = synthesize_insights(top_features, top_n=5)
            insights["reasons"] = synthesized["reasons"]
            insights["suggestions"] = synthesized["suggestions"]

        explanation["insights"] = insights
        explanation["top_features"] = top_features

        # Create SHAP plot image if possible
        shap_plot_base64 = None
        if top_features:
            try:
                feature_names = list(top_features.keys())
                shap_values = np.array(list(top_features.values())).astype(float)
                # create plot
                img_b64 = create_shap_plot(shap_values, feature_names, [model_input.get(fn, 0) for fn in feature_names])
                shap_plot_base64 = img_b64
            except Exception:
                shap_plot_base64 = None

        # Build user-friendly input mapping for response
        user_friendly_input = {}
        # Simple mapping: if field exists in FEATURE_MAPPINGS keys and inverse mapping possible, show display name
        for key, val in model_input.items():
            user_friendly_input[key] = val
            # try to map numeric encoded value to display label if mapping exists
            for fmap_key, fmap in []:  # intentionally empty; we avoid duplicating full mappings in this snippet
                pass

        response = {
            "success": True,
            "prediction": prediction,
            "explanation": explanation,
            "shap_plot": shap_plot_base64,
            "user_input": user_friendly_input,
            "model_input": model_input,
        }

        logging.info("Prediction request handled: prediction=%s", prediction.get("prediction"))
        return jsonify(response)

    except Exception as e:
        logging.exception("Prediction error")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health")
def health():
    if predictor is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500
    return jsonify({"status": "ok", "message": "Service healthy"})


def create_shap_plot(shap_values, feature_names, feature_values):
    """
    Build a horizontal bar SHAP plot and return base64 PNG bytes.
    """
    try:
        plt.figure(figsize=(8, max(2, len(shap_values) * 0.6)))
        shap_abs = np.abs(shap_values)
        order = np.argsort(shap_abs)[::-1]
        ordered_vals = shap_values[order]
        ordered_names = [feature_names[i] for i in order]
        colors = ["#ef4444" if v < 0 else "#2563eb" for v in ordered_vals]
        y_pos = list(range(len(ordered_vals)))
        plt.barh(y_pos, ordered_vals, color=colors, alpha=0.85)
        plt.yticks(y_pos, ordered_names)
        plt.xlabel("SHAP value (impact on prediction)")
        plt.axvline(0, color="#333", alpha=0.2)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        buf.seek(0)
        data = base64.b64encode(buf.read()).decode("utf-8")
        plt.close()
        return data
    except Exception as e:
        logging.exception("Failed to create SHAP plot: %s", e)
        try:
            plt.close()
        except Exception:
            pass
        return None


if __name__ == "__main__":
    # Initialize predictor (load model)
    ok = initialize_predictor()
    if not ok:
        print("Model initialization failed - exiting.")
        raise SystemExit(1)

    # Open browser automatically after server starts (slight delay)
    def _open_browser():
        time.sleep(0.6)
        try:
            webbrowser.open("http://127.0.0.1:5000")
        except Exception:
            pass

    threading.Thread(target=_open_browser, daemon=True).start()

    # Run Flask app (no reloader so model isn't double-loaded)
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)

