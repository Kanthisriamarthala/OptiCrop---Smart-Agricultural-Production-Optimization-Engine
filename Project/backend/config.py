import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(BASE_DIR, "dataset", "Crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "crop_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")

STATIC_DIR = os.path.join(BASE_DIR, "static")
IMAGES_DIR = os.path.join(STATIC_DIR, "images")

SECRET_KEY = os.environ.get("SECRET_KEY", "opticrop-secret-key-change-in-production")
DEBUG = os.environ.get("DEBUG", "False").lower() == "true"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", 5000))

# CORS: allow frontend origin
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")

CROP_INFO = {
    "rice": {
        "description": "Rice is a staple food crop grown in warm, humid conditions with high rainfall. It is primarily cultivated in flooded fields.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "150 – 250 cm",
        "ideal_ph": "5.0 – 7.0",
        "tips": "Ensure proper water management. Use nitrogen-rich fertilizers. Maintain field flooding during growth."
    },
    "maize": {
        "description": "Maize (corn) is a versatile cereal grain that grows in warm temperatures and moderate rainfall conditions.",
        "ideal_temp": "20°C – 32°C",
        "ideal_rainfall": "80 – 180 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Ensure proper spacing. Use balanced NPK fertilizers. Practice crop rotation to maintain soil health."
    },
    "chickpea": {
        "description": "Chickpea is a nutritious legume crop suitable for dry and semi-arid regions with cool temperatures.",
        "ideal_temp": "18°C – 30°C",
        "ideal_rainfall": "60 – 120 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Requires well-drained soil. Avoid excessive irrigation. Good source of protein for soil enrichment."
    },
    "kidneybeans": {
        "description": "Kidney beans are a warm-season legume crop valued for their high protein content.",
        "ideal_temp": "18°C – 28°C",
        "ideal_rainfall": "80 – 150 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Require well-drained loamy soil. Avoid waterlogging. Use phosphorus-rich fertilizers."
    },
    "pigeonpeas": {
        "description": "Pigeonpea is a drought-tolerant legume crop widely grown in tropical and subtropical regions.",
        "ideal_temp": "20°C – 30°C",
        "ideal_rainfall": "100 – 180 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Requires minimal fertilizers. Improves soil nitrogen. Good for intercropping."
    },
    "mothbeans": {
        "description": "Moth bean is a drought-resistant legume crop suitable for arid and semi-arid regions.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "60 – 120 cm",
        "ideal_ph": "6.0 – 8.0",
        "tips": "Thrives in sandy and well-drained soils. Requires low water input. Good for dryland farming."
    },
    "mungbean": {
        "description": "Mung bean is a short-duration legume crop grown for its protein-rich seeds.",
        "ideal_temp": "20°C – 30°C",
        "ideal_rainfall": "100 – 180 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Short growing cycle. Requires warm weather. Good green manure crop."
    },
    "blackgram": {
        "description": "Black gram is a nutritious legume crop widely cultivated in tropical regions.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "100 – 180 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Requires well-drained soil. Responds well to phosphorus. Can be grown in rice fallows."
    },
    "lentil": {
        "description": "Lentil is a cool-season legume crop with high nutritional value.",
        "ideal_temp": "15°C – 25°C",
        "ideal_rainfall": "60 – 120 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Grows best in cool weather. Requires well-drained soil. Avoid excessive moisture."
    },
    "pomegranate": {
        "description": "Pomegranate is a drought-tolerant fruit crop suitable for subtropical and tropical regions.",
        "ideal_temp": "20°C – 30°C",
        "ideal_rainfall": "50 – 100 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Requires hot and dry summers. Well-drained soil essential. Prune regularly for better yield."
    },
    "banana": {
        "description": "Banana is a tropical fruit crop requiring warm temperatures and high humidity.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "150 – 250 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Requires rich, well-drained soil. Heavy feeder of potassium. Provide wind protection."
    },
    "mango": {
        "description": "Mango is a tropical fruit tree that thrives in warm, frost-free climates.",
        "ideal_temp": "25°C – 35°C",
        "ideal_rainfall": "80 – 150 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Requires dry weather during flowering. Deep well-drained soil. Regular pruning needed."
    },
    "grapes": {
        "description": "Grapes are a temperate fruit crop grown for fresh consumption and wine production.",
        "ideal_temp": "20°C – 30°C",
        "ideal_rainfall": "60 – 120 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Require well-drained soil. Trellis support needed. Prune annually for better yield."
    },
    "watermelon": {
        "description": "Watermelon is a warm-season fruit crop requiring long, hot summers.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "60 – 120 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Requires sandy loam soil. Adequate water during fruit development. Control weeds."
    },
    "muskmelon": {
        "description": "Muskmelon is a warm-season fruit crop with sweet, aromatic flesh.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "60 – 120 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Requires warm weather. Well-drained soil. Adequate irrigation during fruit set."
    },
    "apple": {
        "description": "Apple is a temperate fruit crop requiring cold winters and moderate summers.",
        "ideal_temp": "10°C – 22°C",
        "ideal_rainfall": "80 – 150 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Requires chilling hours. Well-drained soil. Regular pruning and pest management."
    },
    "orange": {
        "description": "Orange is a citrus fruit crop grown in subtropical and tropical regions.",
        "ideal_temp": "20°C – 30°C",
        "ideal_rainfall": "80 – 150 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Requires well-drained soil. Regular irrigation. Protect from strong winds."
    },
    "papaya": {
        "description": "Papaya is a tropical fruit crop with rapid growth and high yield potential.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "120 – 220 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Requires rich, well-drained soil. Regular watering. Protect from frost."
    },
    "coconut": {
        "description": "Coconut is a tropical palm crop grown in coastal regions with high humidity.",
        "ideal_temp": "25°C – 35°C",
        "ideal_rainfall": "150 – 250 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Requires sandy coastal soil. High potassium needs. Regular removal of dried leaves."
    },
    "cotton": {
        "description": "Cotton is a fiber crop grown in warm climates with moderate rainfall.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "80 – 180 cm",
        "ideal_ph": "5.5 – 7.5",
        "tips": "Requires deep well-drained soil. Pest management crucial. Harvest after boll opening."
    },
    "coffee": {
        "description": "Coffee is a tropical beverage crop grown in shaded, high-altitude conditions.",
        "ideal_temp": "18°C – 28°C",
        "ideal_rainfall": "120 – 200 cm",
        "ideal_ph": "5.0 – 6.5",
        "tips": "Requires shade. Acidic soil preferred. Regular pruning and mulching beneficial."
    },
    "jute": {
        "description": "Jute is a natural fiber crop grown in warm, humid conditions with high rainfall.",
        "ideal_temp": "22°C – 32°C",
        "ideal_rainfall": "150 – 250 cm",
        "ideal_ph": "5.5 – 7.0",
        "tips": "Requires alluvial soil. High water requirement. Retting process needed for fiber extraction."
    }
}
