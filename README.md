Insurance-Cross-Sell-Prediction

This project demonstrates MLOps best practices with a simple machine learning model. It includes:

- ML model training with scikit-learn
- MLflow for experiment tracking
- Flask API for model serving
- Docker containerization
- CI/CD with GitHub Actions
- Unit tests with pytest

## Project Structure
```
├── src/
│   ├── train.py      # Model training script
│   └── predict.py    # Flask API for predictions
├── tests/
│   └── test_model.py # Unit tests
├── models/           # Saved models
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md

## Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mlops-project.git
cd mlops-project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```
## Running the Project

1. Train the model:
```bash
python src/train.py
```

2. Run the prediction service:
```bash
python src/predict.py
```

3. Run with Docker:
```bash
docker-compose up
```
## Making Predictions

Send POST requests to the prediction endpoint:
```bash
curl -X POST http://localhost:5000/predict \
-H "Content-Type: application/json" \
-d '{"features": [5.1, 3.5, 1.4, 0.2]}'
```

## Running Tests

```bash
pytest tests/
```
## MLflow Tracking

Access MLflow UI at http://localhost:5000 to view experiment tracking results.
