# OptiCrop - Smart Agricultural Production Optimization Engine

## Project Structure

```
OptiCrop/
├── frontend/          # Static site — deploy on Vercel
│   ├── index.html     # Home page
│   ├── recommend.html # Crop recommendation (JS → API)
│   ├── suitability.html # Suitability analysis (JS → API)
│   ├── dashboard.html # Research dashboard (JS → API)
│   ├── about.html     # About page
│   ├── contact.html   # Contact page
│   ├── css/style.css  # Styles
│   ├── js/script.js   # Frontend scripts
│   ├── images/        # Chart images (fallback)
│   └── vercel.json    # Vercel deployment config
│
├── backend/           # Flask API — deploy on Render
│   ├── app.py         # REST API server (Flask)
│   ├── config.py      # Configuration
│   ├── requirements.txt
│   ├── Procfile       # Render start command
│   ├── dataset/Crop_recommendation.csv
│   ├── model/
│   │   ├── predict.py      # Prediction engine
│   │   ├── train_model.py  # Model training
│   │   ├── crop_model.pkl  # Trained RF model
│   │   ├── label_encoder.pkl
│   │   ├── metrics.json
│   │   └── pca_data.pkl
│   ├── utils/
│   │   ├── preprocessing.py
│   │   ├── validation.py
│   │   └── visualization.py
│   └── static/images/  # Generated chart images
│
├── (original files)    # Legacy monolithic version
├── deploy-config.json
└── README.md
```

## API Endpoints (Backend)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/predict` | Crop prediction |
| POST | `/api/suitability` | Suitability assessment |
| GET | `/api/dashboard` | Dashboard data (metrics, stats, charts) |
| GET | `/api/metrics` | Model comparison metrics |
| GET | `/api/crops` | List of all crop names |
| GET | `/api/crop-info/<crop>` | Info for a specific crop |
| GET | `/api/images/<filename>` | Chart image file |

## Deploy

### Backend (Render)
```
cd backend
pip install -r requirements.txt
gunicorn app:app --bind 0.0.0.0:$PORT
```

### Frontend (Vercel)
```
cd frontend
vercel
```

Update `API_BASE` in frontend JS files to point to your Render backend URL before deploying.
