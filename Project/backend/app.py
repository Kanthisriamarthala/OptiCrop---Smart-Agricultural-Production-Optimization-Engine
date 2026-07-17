import os
import sys
import json
import traceback
import logging
import numpy as np

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SECRET_KEY, DEBUG, HOST, PORT, CROP_INFO, METRICS_PATH, IMAGES_DIR, FRONTEND_URL
from utils.validation import validate_prediction_input
from utils.visualization import get_existing_chart_paths, charts_exist_on_disk, generate_all_charts, CHART_FILES
from utils.preprocessing import load_dataset
from model.predict import predict_crop, assess_suitability, load_model, load_encoder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app, resources={r"/api/*": {"origins": "*"}})

CHART_PATHS = {}
DASHBOARD_STATS = {}


def compute_dashboard_stats():
    global DASHBOARD_STATS
    if DASHBOARD_STATS:
        return DASHBOARD_STATS
    try:
        df = load_dataset()
        stats = {}
        stats["total_records"] = len(df)
        stats["total_crops"] = df["label"].nunique()
        stats["features"] = list(df.select_dtypes(include=[np.number]).columns)
        stats["avg_temperature"] = round(df["temperature"].mean(), 1)
        stats["avg_humidity"] = round(df["humidity"].mean(), 1)
        stats["avg_rainfall"] = round(df["rainfall"].mean(), 1)
        stats["avg_ph"] = round(df["ph"].mean(), 2)
        stats["avg_n"] = round(df["N"].mean(), 1)
        stats["avg_p"] = round(df["P"].mean(), 1)
        stats["avg_k"] = round(df["K"].mean(), 1)
        stats["min_rainfall"] = round(df["rainfall"].min(), 1)
        stats["max_rainfall"] = round(df["rainfall"].max(), 1)
        stats["min_temp"] = round(df["temperature"].min(), 1)
        stats["max_temp"] = round(df["temperature"].max(), 1)
        stats["most_frequent_crop"] = df["label"].mode()[0]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr = df[numeric_cols].corr()
        triu = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_pairs = []
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                if triu[i, j]:
                    corr_pairs.append({
                        "pair": f"{corr.columns[i]} vs {corr.columns[j]}",
                        "value": round(corr.iloc[i, j], 3)
                    })
        corr_pairs.sort(key=lambda x: abs(x["value"]), reverse=True)
        stats["top_correlations"] = corr_pairs[:5]

        DASHBOARD_STATS = stats
        logger.info("Dashboard stats computed successfully.")
        return stats
    except Exception as e:
        logger.error(f"Failed to compute dashboard stats: {e}")
        return {}


def load_or_generate_charts():
    global CHART_PATHS
    if CHART_PATHS:
        return CHART_PATHS
    if charts_exist_on_disk():
        CHART_PATHS = get_existing_chart_paths()
        if CHART_PATHS:
            logger.info("Loaded existing chart paths from disk.")
            return CHART_PATHS
    try:
        df = load_dataset()
        CHART_PATHS = generate_all_charts(df)
    except Exception as e:
        logger.warning(f"Could not generate charts: {e}")
        CHART_PATHS = {}


def get_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


def check_model_ready():
    model_path = os.path.join(os.path.dirname(__file__), "model", "crop_model.pkl")
    encoder_path = os.path.join(os.path.dirname(__file__), "model", "label_encoder.pkl")
    return os.path.exists(model_path) and os.path.exists(encoder_path)


# ----- API Routes -----

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "model_ready": check_model_ready()})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json(force=True)
        nitrogen = float(data.get("nitrogen", 0))
        phosphorous = float(data.get("phosphorous", 0))
        potassium = float(data.get("potassium", 0))
        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        ph = float(data.get("ph", 0))
        rainfall = float(data.get("rainfall", 0))
    except (ValueError, TypeError, KeyError):
        return jsonify({"error": "Invalid input data"}), 400

    if not check_model_ready():
        return jsonify({"error": "Model not trained yet"}), 503

    try:
        crop, confidence = predict_crop(
            nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall
        )
        crop_info = CROP_INFO.get(crop, {})
        return jsonify({
            "crop": crop,
            "confidence": confidence,
            "crop_info": crop_info
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/suitability", methods=["POST"])
def api_suitability():
    try:
        data = request.get_json(force=True)
        nitrogen = float(data.get("nitrogen", 0))
        phosphorous = float(data.get("phosphorous", 0))
        potassium = float(data.get("potassium", 0))
        temperature = float(data.get("temperature", 0))
        humidity = float(data.get("humidity", 0))
        ph = float(data.get("ph", 0))
        rainfall = float(data.get("rainfall", 0))
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid input data"}), 400

    errors = validate_prediction_input(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    if not check_model_ready():
        return jsonify({"error": "Model not trained yet"}), 503

    try:
        assessment = assess_suitability(
            nitrogen, phosphorous, potassium, temperature, humidity, ph, rainfall
        )
        return jsonify(assessment)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dashboard", methods=["GET"])
def api_dashboard():
    load_or_generate_charts()
    metrics = get_metrics()
    stats = compute_dashboard_stats()
    chart_list = {}
    for key, rel_path in CHART_PATHS.items():
        chart_list[key] = f"/api/images/{rel_path.replace('images/', '')}"
    return jsonify({
        "metrics": metrics,
        "stats": stats,
        "charts": chart_list
    })


@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    metrics = get_metrics()
    if metrics is None:
        return jsonify({"error": "No metrics available"}), 404
    return jsonify(metrics)


@app.route("/api/crop-info/<crop_name>", methods=["GET"])
def api_crop_info(crop_name):
    info = CROP_INFO.get(crop_name.lower())
    if info is None:
        return jsonify({"error": f"No info available for {crop_name}"}), 404
    return jsonify(info)


@app.route("/api/images/<filename>", methods=["GET"])
def api_chart_image(filename):
    if filename in CHART_FILES or os.path.exists(os.path.join(IMAGES_DIR, filename)):
        return send_from_directory(IMAGES_DIR, filename)
    return jsonify({"error": "Image not found"}), 404


@app.route("/api/crops", methods=["GET"])
def api_crops():
    return jsonify({"crops": list(CROP_INFO.keys())})


if __name__ == "__main__":
    print("Starting OptiCrop Backend API Server")
    print(f"Debug mode: {DEBUG}")
    print(f"Server: http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=DEBUG)
