# agents/agent5_scorer.py
from db.database import get_lead_by_id, update_lead_status
from datetime import datetime
import logging
from typing import Dict
import pandas as pd
import joblib
import os
import shap
import numpy as np

logger = logging.getLogger(__name__)

# === MODEL PATH ===
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml_models", "lead_score_pipeline.pkl")
MODEL_PATH = os.path.abspath(MODEL_PATH)

# --- GLOBAL MODEL AND EXPLAINER ---
ml_model = None
explainer = None

try:
    if os.path.exists(MODEL_PATH):
        ml_model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Agent 5: ML Model Loaded from {MODEL_PATH}")
    else:
        logger.error(f"❌ ML Model Missing at {MODEL_PATH}. Fallback scoring will be used.")
except Exception as e:
    logger.error(f"❌ Failed loading ML Model: {e}")


def determine_priority(score: float) -> str:
    """Priority tag classification based on scaled score."""
    if score >= 80:
        return "Hot"
    elif score >= 50:
        return "Warm"
    return "Cold"


def build_feature_row_from_lead(lead: Dict) -> pd.DataFrame:
    try:
        personal = lead.get("personal", {})
        enrollment = lead.get("enrollment", {})
        interaction = lead.get("interaction", {})
        analysis = lead.get("analysis", {})

        row = {
            "personal.source": personal.get("source", "Other"),
            "personal.location": personal.get("location", "Unknown"),
            "enrollment.course_interest": enrollment.get("course_interest", "General"),
            "interaction.call_count": float(interaction.get("call_count", 0) or 0),
            "interaction.call_duration": float(interaction.get("call_duration", 0) or 0),
            "analysis.sentiment": analysis.get("sentiment", "neutral"),
            "analysis.next_steps": analysis.get("next_steps", "follow-up later"),
        }

        return pd.DataFrame([row])
    except Exception as e:
        logger.error(f"❌ Error building dataframe row: {e}")
        raise


def run_lead_scoring(lead_id: str) -> Dict:
    logger.info(f"🟡 Agent 5: Starting scoring for Lead {lead_id}")

    if ml_model is None:
        logger.warning("⚠ No ML model, using fallback scoring.")
        return {"success": True, "score": 50.0, "tag": "Warm"}

    lead = get_lead_by_id(lead_id)
    if not lead:
        return {"success": False, "error": f"Lead {lead_id} not found."}

    current_status = lead.get("interaction", {}).get("call_status", "")
    if current_status not in ["Follow-up Sent - Complete"]:
        return {"success": False, "error": "Lead is not ready for scoring stage yet."}

    try:
        X = build_feature_row_from_lead(lead)

        raw = float(ml_model.predict(X)[0])
        base_score = raw * 100

        if base_score < 5:
            score = base_score * 6
        elif base_score < 20:
            score = base_score * 3
        elif base_score < 50:
            score = 40 + (base_score - 20)
        else:
            score = 70 + ((base_score - 50) * 1.5)

        # Add CRM manual boost
        call_count = lead.get("interaction", {}).get("call_count", 0)
        score += min(call_count / 2, 20)  # Boost up to +20

        score = round(min(max(score, 0), 100), 3)
        tag = determine_priority(score)

        update_payload = {
            "$set": {
                "score.current_score": score,
                "score.priority_tag": tag,
                "interaction.last_activity": datetime.now(),
                "interaction.call_status": f"Scored ({tag}) - Cycle Complete"
            }
        }

        result = update_lead_status(lead_id, update_payload)

        if result.get("acknowledged"):
            logger.info(f"🏁 Scoring Complete → Lead {lead_id}: Score={score}, Tag={tag}")
            return {"success": True, "score": score, "tag": tag}

        return {"success": False, "error": "DB update failed"}

    except Exception as e:
        logger.error(f"🔥 Score Prediction Failed: {e}")
        return {"success": False, "error": str(e)}

def explain_lead_score(lead_id: str):
    global explainer

    try:
        lead = get_lead_by_id(lead_id)
        if not lead:
            return {"success": False, "error": f"Lead {lead_id} not found"}

        status = lead.get("interaction", {}).get("call_status", "")
        if "Scored" not in status:
            return {"success": False, "error": "Lead must be scored before explanation"}

        # Build ML input row
        X = build_feature_row_from_lead(lead)

        preprocessor = ml_model.named_steps["preprocess"]
        model = ml_model.named_steps["model"]

        # Transform for SHAP
        X_processed = preprocessor.transform(X)
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray().astype(float)

        # Initialize SHAP analyzer
        if explainer is None:
            explainer = shap.TreeExplainer(model)

        shap_values = explainer.shap_values(X_processed)[0]
        base_value = float(explainer.expected_value)
        # --- recreate correct feature list matching SHAP output ---
        ohe_features = preprocessor.transformers_[1][1].get_feature_names_out([
            "personal.source",
            "personal.location",
            "enrollment.course_interest",
            "analysis.sentiment",
            "analysis.next_steps"
        ])

        feature_names = ["interaction.call_count", "interaction.call_duration"] + list(ohe_features)

        # Ensure same length
        if len(feature_names) != len(shap_values):
            raise ValueError(f"Feature mismatch: {len(feature_names)} vs SHAP {len(shap_values)}")



        ranked = sorted(zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True)

        # === SAME scoring logic from run_lead_scoring() ===
        raw_pred = float(ml_model.predict(X)[0])
        base_score = raw_pred * 100

        if base_score < 5:
            score = base_score * 6
        elif base_score < 20:
            score = base_score * 3
        elif base_score < 50:
            score = 40 + (base_score - 20)
        else:
            score = 70 + ((base_score - 50) * 1.5)

        call_count = lead.get("interaction", {}).get("call_count", 0)
        score += min(call_count / 2, 20)

        score = round(min(max(score, 0), 100), 3)

        return {
            "success": True,
            "prediction": score,
            "tag": determine_priority(score),
            "feature_impacts": [
                {"feature": fn, "impact": round(float(v), 3)} for fn, v in ranked
            ],
            "shap_values": shap_values.tolist(),
            "base_value": base_value,
            "feature_values": X_processed[0].tolist(),
            "feature_names": feature_names
        }

    except Exception as e:
        logger.error(f"❌ XAI Failure: {e}")
        return {"success": False, "error": str(e)}
