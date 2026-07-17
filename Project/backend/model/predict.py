import os
import sys
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import MODEL_PATH, ENCODER_PATH

FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train_model.py first."
        )
    model = joblib.load(MODEL_PATH)
    return model


def load_encoder():
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(
            f"Encoder not found at {ENCODER_PATH}. Run train_model.py first."
        )
    encoder = joblib.load(ENCODER_PATH)
    return encoder


def predict_crop(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall):
    model = load_model()
    encoder = load_encoder()
    features = pd.DataFrame(
        [[nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall]],
        columns=FEATURE_COLUMNS,
    )
    pred_encoded = model.predict(features)
    crop_name = encoder.inverse_transform(pred_encoded)[0]
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)
        confidence = float(np.max(proba))
    else:
        confidence = None
    return crop_name, confidence


def assess_suitability(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall):
    from config import CROP_INFO
    crop, confidence = predict_crop(nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall)
    info = CROP_INFO.get(crop, {})
    temp_range = info.get("ideal_temp", "")
    rainfall_range = info.get("ideal_rainfall", "")
    ph_range = info.get("ideal_ph", "")

    suitability_status = "Optimal"
    issues = []

    if temp_range:
        try:
            parts = temp_range.replace("°C", "").split("–")
            low, high = float(parts[0].strip()), float(parts[1].strip())
            margin = (high - low) * 0.2
            if temperature < low - margin or temperature > high + margin:
                suitability_status = "Poor"
                issues.append(f"Temperature {temperature}°C is outside ideal range {temp_range}.")
            elif temperature < low or temperature > high:
                suitability_status = "Moderate"
                issues.append(f"Temperature {temperature}°C is at the edge of the ideal range {temp_range}.")
        except (ValueError, IndexError):
            pass

    if rainfall_range:
        try:
            parts = rainfall_range.replace(" cm", "").split("–")
            low, high = float(parts[0].strip()), float(parts[1].strip())
            margin = (high - low) * 0.2
            if rainfall < low - margin or rainfall > high + margin:
                if suitability_status != "Poor":
                    suitability_status = "Poor"
                issues.append(f"Rainfall {rainfall} cm is outside ideal range {rainfall_range}.")
            elif rainfall < low or rainfall > high:
                if suitability_status != "Poor":
                    suitability_status = "Moderate"
                issues.append(f"Rainfall {rainfall} cm is at the edge of the ideal range {rainfall_range}.")
        except (ValueError, IndexError):
            pass

    if ph_range:
        try:
            parts = ph_range.split(" – ")
            low, high = float(parts[0].strip()), float(parts[1].strip())
            margin = (high - low) * 0.2
            if ph < low - margin or ph > high + margin:
                if suitability_status != "Poor":
                    suitability_status = "Poor"
                issues.append(f"pH {ph} is outside ideal range {ph_range}.")
            elif ph < low or ph > high:
                if suitability_status != "Poor":
                    suitability_status = "Moderate"
                issues.append(f"pH {ph} is at the edge of the ideal range {ph_range}.")
        except (ValueError, IndexError):
            pass

    n_optimal = nitrogen > 50
    p_optimal = phosphorous > 30
    k_optimal = potassium > 40
    soil_status = []
    if n_optimal:
        soil_status.append("Nitrogen: Optimal")
    else:
        soil_status.append("Nitrogen: Low \u2014 consider nitrogen-rich fertilizers")
    if p_optimal:
        soil_status.append("Phosphorous: Optimal")
    else:
        soil_status.append("Phosphorous: Low \u2014 consider phosphate fertilizers")
    if k_optimal:
        soil_status.append("Potassium: Optimal")
    else:
        soil_status.append("Potassium: Low \u2014 consider potash fertilizers")

    env_compatibility = {
        "Temperature": "Compatible" if not any("Temperature" in i for i in issues) else "Incompatible",
        "Humidity": "Compatible" if 60 <= humidity <= 90 else "Suboptimal",
        "Rainfall": "Compatible" if not any("Rainfall" in i for i in issues) else "Incompatible",
        "pH": "Compatible" if not any("pH" in i for i in issues) else "Incompatible",
    }

    if suitability_status == "Optimal":
        productivity = "High \u2014 All conditions are favorable for maximum yield."
    elif suitability_status == "Moderate":
        productivity = "Moderate \u2014 Some conditions are suboptimal, potentially reducing yield."
    else:
        productivity = "Low \u2014 Multiple conditions are unfavorable, significant yield reduction expected."

    if rainfall < 60:
        rainfall_analysis = "Low rainfall conditions \u2014 consider supplemental irrigation."
    elif 60 <= rainfall <= 120:
        rainfall_analysis = "Moderate rainfall \u2014 suitable for rainfed agriculture."
    else:
        rainfall_analysis = "High rainfall \u2014 ensure proper drainage and flood management."

    if temperature < 15:
        temp_analysis = "Cool conditions \u2014 suitable for temperate crops."
    elif 15 <= temperature <= 25:
        temp_analysis = "Mild conditions \u2014 suitable for a wide range of crops."
    elif 25 <= temperature <= 35:
        temp_analysis = "Warm conditions \u2014 ideal for tropical and subtropical crops."
    else:
        temp_analysis = "Hot conditions \u2014 select heat-tolerant varieties."

    if suitability_status == "Optimal":
        recommendation = (
            f"Excellent choice! {crop.title()} is highly suitable for your current conditions. "
            f"Follow best agricultural practices for optimal yield."
        )
    elif suitability_status == "Moderate":
        recommendation = (
            f"{crop.title()} can be grown with some adjustments. "
            f"Address the highlighted issues to improve suitability."
        )
    else:
        recommendation = (
            f"{crop.title()} may not be the most suitable crop for current conditions. "
            f"Consider soil amendment or selecting a different crop."
        )

    productivity_level = "High" if suitability_status == "Optimal" else ("Medium" if suitability_status == "Moderate" else "Low")

    return {
        "crop": crop,
        "confidence": confidence,
        "suitability_status": suitability_status,
        "productivity": productivity,
        "productivity_level": productivity_level,
        "soil_analysis": soil_status,
        "environmental_compatibility": env_compatibility,
        "rainfall_analysis": rainfall_analysis,
        "temperature_analysis": temp_analysis,
        "recommendation": recommendation,
        "issues": issues,
        "crop_info": info,
    }
