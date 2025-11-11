"""
AQI Alert Prediction Model
Predicts air quality alert severity for personalized citizen alerts
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from typing import Dict, Tuple, Any
import shap

class AQIAlertModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'pm25', 'pm10', 'no2', 'so2', 'co', 'o3',
            'pm25_1h_mean', 'pm25_6h_mean', 'pm25_24h_mean',
            'pm25_pm10_ratio', 'pm25_change_pct',
            'hour', 'day_of_week', 'is_weekend',
            'temperature', 'humidity', 'wind_speed',
            'user_age', 'has_health_condition'
        ]

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for AQI prediction"""
        df = data.copy()

        # Rolling statistics
        if 'pm25' in df.columns:
            df['pm25_1h_mean'] = df['pm25'].rolling(window=1, min_periods=1).mean()
            df['pm25_6h_mean'] = df['pm25'].rolling(window=6, min_periods=1).mean()
            df['pm25_24h_mean'] = df['pm25'].rolling(window=24, min_periods=1).mean()

            # Pollutant ratios
            df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1)

            # Change detection
            df['pm25_change_pct'] = df['pm25'].pct_change().fillna(0) * 100

        # Temporal features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Fill missing values
        df = df.fillna(df.mean())

        return df

    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train the AQI alert classification model"""
        # Feature engineering
        X_features = self.create_features(X)

        # Use only available features
        available_features = [f for f in self.feature_names if f in X_features.columns]
        X_train = X_features[available_features]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        self.model.fit(X_scaled, y)

        # Training metrics
        train_score = self.model.score(X_scaled, y)

        return {
            "train_accuracy": train_score,
            "n_features": len(available_features),
            "features": available_features
        }

    def predict(self, features: Dict[str, float]) -> Tuple[str, float, Dict[str, Any]]:
        """
        Predict AQI alert level
        Returns: (alert_level, probability, explanation)
        """
        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Feature engineering
        df_features = self.create_features(df)

        # Extract features in correct order
        available_features = [f for f in self.feature_names if f in df_features.columns]
        X = df_features[available_features].values

        # Scale
        X_scaled = self.scaler.transform(X)

        # Predict
        probabilities = self.model.predict_proba(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]

        # Map to alert levels
        alert_levels = ['LOW', 'MODERATE', 'HIGH', 'SEVERE']
        alert_level = alert_levels[prediction] if prediction < len(alert_levels) else 'SEVERE'

        # Generate explanation
        explanation = self._generate_explanation(features, X_scaled, prediction)

        return alert_level, float(probabilities[prediction]), explanation

    def _generate_explanation(self, features: Dict, X_scaled: np.ndarray, prediction: int) -> Dict[str, Any]:
        """Generate human-readable explanation for prediction"""
        explanation = {
            "reason": "",
            "key_factors": []
        }

        # Feature importance based explanation
        if self.model:
            feature_importance = self.model.feature_importances_
            top_features_idx = np.argsort(feature_importance)[-3:]

            available_features = [f for f in self.feature_names if f in features]
            for idx in top_features_idx:
                if idx < len(available_features):
                    feat_name = available_features[idx]
                    explanation["key_factors"].append({
                        "feature": feat_name,
                        "importance": float(feature_importance[idx])
                    })

        # Rule-based reasons
        pm25 = features.get('pm25', 0)
        pm25_change = features.get('pm25_change_pct', 0)

        if pm25 > 300:
            explanation["reason"] = f"PM2.5 level critically high at {pm25:.1f}"
        elif pm25 > 150:
            explanation["reason"] = f"PM2.5 level elevated at {pm25:.1f}"
        elif abs(pm25_change) > 50:
            explanation["reason"] = f"PM2.5 changed by {pm25_change:.1f}% in last hour"
        else:
            explanation["reason"] = "Air quality conditions assessed based on multiple factors"

        return explanation

    def save(self, model_path: str, scaler_path: str):
        """Save model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path: str, scaler_path: str):
        """Load model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)


def generate_synthetic_aqi_data(n_samples: int = 1000) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic AQI data for training"""
    np.random.seed(42)

    data = {
        'pm25': np.random.exponential(50, n_samples),
        'pm10': np.random.exponential(80, n_samples),
        'no2': np.random.exponential(30, n_samples),
        'so2': np.random.exponential(20, n_samples),
        'co': np.random.exponential(1, n_samples),
        'o3': np.random.exponential(40, n_samples),
        'temperature': np.random.normal(25, 5, n_samples),
        'humidity': np.random.uniform(30, 90, n_samples),
        'wind_speed': np.random.exponential(10, n_samples),
        'user_age': np.random.randint(18, 80, n_samples),
        'has_health_condition': np.random.binomial(1, 0.3, n_samples),
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='H')
    }

    df = pd.DataFrame(data)

    # Generate labels based on PM2.5 levels
    labels = []
    for pm25 in df['pm25']:
        if pm25 < 50:
            labels.append(0)  # LOW
        elif pm25 < 100:
            labels.append(1)  # MODERATE
        elif pm25 < 200:
            labels.append(2)  # HIGH
        else:
            labels.append(3)  # SEVERE

    return df, np.array(labels)


if __name__ == "__main__":
    # Train and save model
    print("Training AQI Alert Model...")
    X, y = generate_synthetic_aqi_data(2000)

    model = AQIAlertModel()
    metrics = model.train(X, y)
    print(f"Training completed: {metrics}")

    # Save model
    model.save("models/aqi_alert_model.pkl", "models/aqi_alert_scaler.pkl")
    print("Model saved successfully!")

    # Test prediction
    test_input = {
        'pm25': 180, 'pm10': 250, 'no2': 45, 'so2': 25, 'co': 1.5, 'o3': 50,
        'pm25_change_pct': 60, 'temperature': 28, 'humidity': 65, 'wind_speed': 8,
        'user_age': 35, 'has_health_condition': 1
    }
    alert, prob, explanation = model.predict(test_input)
    print(f"\nTest Prediction: {alert} (confidence: {prob:.2f})")
    print(f"Explanation: {explanation}")
