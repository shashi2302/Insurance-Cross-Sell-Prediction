import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data():
    """Load sample iris dataset"""
    from sklearn.datasets import load_iris
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df

def train_model():
    """Train the model and log metrics with MLflow"""
    logger.info("Starting model training...")
    
    # Load and split data
    data = load_data()
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        # Log parameters
        mlflow.log_param("n_estimators", 100)
        
        # Log metrics
        train_score = rf.score(X_train, y_train)
        test_score = rf.score(X_test, y_test)
        mlflow.log_metric("train_accuracy", train_score)
        mlflow.log_metric("test_accuracy", test_score)
        
        # Save model
        joblib.dump(rf, 'models/model.joblib')
        mlflow.sklearn.log_model(rf, "random_forest_model")
        
        logger.info(f"Model training completed. Test accuracy: {test_score}")
        
if __name__ == "__main__":
    train_model()