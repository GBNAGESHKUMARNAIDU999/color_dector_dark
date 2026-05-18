# 🎨 ColorAI — AI-Powered Real-Time Color Detection & Tracking System

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9-green)](https://opencv.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)](https://streamlit.io)

A production-grade, enterprise-level AI Color Detection and Object Tracking Platform.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit Frontend (recommended)
```bash
streamlit run frontend/app.py
```
Open → http://localhost:8501

### 3. Run FastAPI Backend (optional, for API access)
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
API Docs → http://localhost:8000/docs

### 4. Run with Docker
```bash
docker-compose up --build
```

---

## 🗂️ Project Structure

```
color detector ml/
├── frontend/           # Streamlit multi-page UI
│   ├── app.py         # Dashboard home page
│   ├── pages/         # 4 feature pages
│   └── components/    # Reusable UI components
├── backend/            # FastAPI REST API
│   ├── main.py
│   ├── routers/
│   ├── schemas/
│   └── services/
├── cv_pipeline/        # Core computer vision
│   ├── color_detector.py
│   ├── color_classifier.py
│   ├── color_spaces.py
│   ├── preprocessor.py
│   └── segmentation.py
├── tracking/           # Object tracking
│   ├── contour_tracker.py
│   └── object_tracker.py
├── ml/                 # Machine learning
│   ├── kmeans_clustering.py
│   ├── color_similarity.py
│   └── color_recommender.py
├── datasets/
│   └── colors.csv      # 200+ named colors
├── tests/              # Full test suite
├── deployment/         # Dockerfiles
└── utils/              # Config, logging, validation
```

---

## ✨ Features

### 🖼️ Image Color Detection
- Upload any image (JPG, PNG, BMP, WebP)
- Automatic multi-color detection with bounding boxes
- K-Means dominant color extraction (up to 12 colors)
- RGB, HSV, HEX, LAB color space display
- 200+ named color lookup (nearest neighbor)
- Color harmony palettes (complementary, triadic, analogous, etc.)

### 📷 Real-Time Webcam Detection
- Live webcam feed via browser (streamlit-webrtc)
- Multi-color simultaneous detection and tracking
- Centroid-based object tracking with movement trails
- FPS counter and live confidence display

### 🎬 Video Analysis
- Frame-by-frame color detection timeline
- Scene-level dominant color aggregation
- Plotly interactive timeline chart
- Downloadable CSV and JSON reports

### 🤖 Advanced AI Features
- **Fashion Color Matcher** — warm/cool/neutral tone detection + outfit palettes
- **Interior Design Recommender** — mood-based room color palettes
- **Traffic Light Detector** — RED/YELLOW/GREEN state detection
- **Fruit Ripeness Analyzer** — hue-based ripeness estimation
- **Full Palette Studio** — all 6 harmony types at once

---

## 🔬 Tech Stack

| Layer | Technology |
|-------|-----------|
| **CV Core** | OpenCV 4.9, NumPy |
| **ML** | Scikit-learn (K-Means), SciPy (Delta-E 2000) |
| **Backend API** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | Streamlit, streamlit-webrtc, Plotly |
| **Logging** | Loguru (rotating, colored) |
| **Testing** | Pytest, httpx (async API tests) |
| **Deployment** | Docker, Docker Compose |

---

## 🧪 Running Tests
```bash
pytest tests/ -v --tb=short
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Health check |
| `POST` | `/api/image/analyze` | Full image color analysis |
| `GET`  | `/api/image/color-info?r=&g=&b=` | Color space info for RGB value |
| `POST` | `/api/video/analyze` | Video frame-by-frame analysis |

Full Swagger docs: http://localhost:8000/docs

---

## 🐳 Docker Deployment
```bash
# Build and run both services
docker-compose up --build

# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

---

## 🎨 Color Detection Details

**Supported Detection Colors:**
Red, Orange, Yellow, Green, Cyan, Blue, Purple, Pink, White, Black

**Color Spaces Supported:**
- RGB (0-255 per channel)
- HSV (OpenCV scale: H 0-179, S/V 0-255)
- HEX (#RRGGBB)
- CIE LAB (perceptually uniform)

**Named Color Database:** 200+ colors with family classification

---

## 📸 Screenshots

| Dashboard | Image Detection | Video Analysis | Advanced AI |
|-----------|----------------|----------------|-------------|
| Home page with feature cards | Bounding boxes + palette | Color timeline chart | Fashion + interior tools |
