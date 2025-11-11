"""
Traffic/Route ETA Prediction Model
Predicts traffic delays and route ETAs based on historical patterns and current conditions
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
from typing import Dict, Tuple, Any, List
from datetime import datetime

class TrafficETAModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = [
            'hour', 'day_of_week', 'is_weekend', 'is_rush_hour',
            'temperature', 'rain_mm', 'visibility_km',
            'historical_avg_speed', 'segment_length_km',
            'num_traffic_lights', 'is_highway', 'incident_nearby'
        ]

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for traffic prediction"""
        df = data.copy()

        # Temporal features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

            # Rush hour detection
            df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) |
                                   (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)

        # Weather impact
        if 'rain_mm' in df.columns:
            df['rain_mm'] = df['rain_mm'].fillna(0)

        # Fill missing values
        df = df.fillna(df.mean())

        return df

    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train the traffic ETA prediction model"""
        # Feature engineering
        X_features = self.create_features(X)

        # Use only available features
        available_features = [f for f in self.feature_names if f in X_features.columns]
        X_train = X_features[available_features]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.model.fit(X_scaled, y)

        # Training metrics
        train_score = self.model.score(X_scaled, y)
        predictions = self.model.predict(X_scaled)
        mae = np.mean(np.abs(predictions - y))

        return {
            "train_r2_score": train_score,
            "train_mae_minutes": mae,
            "n_features": len(available_features),
            "features": available_features
        }

    def predict(self, features: Dict[str, float]) -> Tuple[float, float, Dict[str, Any]]:
        """
        Predict traffic delay in minutes
        Returns: (delay_minutes, confidence, explanation)
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
        delay_minutes = float(self.model.predict(X_scaled)[0])

        # Calculate confidence (based on tree variance)
        predictions = np.array([tree.predict(X_scaled)[0] for tree in self.model.estimators_])
        confidence = 1.0 - (np.std(predictions) / (np.mean(predictions) + 1))
        confidence = max(0.0, min(1.0, confidence))

        # Generate explanation
        explanation = self._generate_explanation(features, delay_minutes)

        return delay_minutes, float(confidence), explanation

    def predict_route_eta(self, segments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Predict ETA for entire route by summing segment predictions"""
        total_delay = 0
        total_distance = 0
        segment_details = []

        for segment in segments:
            delay, confidence, explanation = self.predict(segment)
            total_delay += delay
            total_distance += segment.get('segment_length_km', 0)

            segment_details.append({
                "segment_id": segment.get('segment_id', 'unknown'),
                "delay_minutes": round(delay, 2),
                "confidence": round(confidence, 2),
                "reason": explanation.get('reason', '')
            })

        # Calculate total ETA
        base_time = (total_distance / 40) * 60  # Assuming 40 km/h base speed
        total_eta = base_time + total_delay

        return {
            "total_eta_minutes": round(total_eta, 2),
            "total_delay_minutes": round(total_delay, 2),
            "total_distance_km": round(total_distance, 2),
            "segments": segment_details
        }

    def _generate_explanation(self, features: Dict, delay_minutes: float) -> Dict[str, Any]:
        """Generate human-readable explanation for prediction"""
        explanation = {
            "reason": "",
            "key_factors": []
        }

        # Rule-based reasons
        is_rush_hour = features.get('is_rush_hour', 0)
        rain = features.get('rain_mm', 0)
        incident = features.get('incident_nearby', 0)
        is_highway = features.get('is_highway', 0)

        reasons = []
        if incident:
            reasons.append("incident reported nearby")
        if is_rush_hour:
            reasons.append("rush hour traffic")
        if rain > 5:
            reasons.append(f"heavy rain ({rain:.1f}mm)")
        if delay_minutes > 15 and not is_highway:
            reasons.append("high congestion on local roads")

        if reasons:
            explanation["reason"] = f"Delay due to: {', '.join(reasons)}"
        else:
            explanation["reason"] = "Normal traffic conditions"

        # Feature importance
        if self.model:
            feature_importance = self.model.feature_importances_
            available_features = [f for f in self.feature_names if f in features]
            for i, feat in enumerate(available_features[:3]):
                if i < len(feature_importance):
                    explanation["key_factors"].append({
                        "feature": feat,
                        "value": features.get(feat, 0),
                        "importance": float(feature_importance[i])
                    })

        return explanation

    def save(self, model_path: str, scaler_path: str):
        """Save model and scaler"""
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path: str, scaler_path: str):
        """Load model and scaler"""
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)


def generate_synthetic_traffic_data(n_samples: int = 2000) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic traffic data for training"""
    np.random.seed(42)

    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='15min')

    data = {
        'timestamp': timestamps,
        'temperature': np.random.normal(25, 8, n_samples),
        'rain_mm': np.random.exponential(2, n_samples),
        'visibility_km': np.random.normal(10, 3, n_samples),
        'historical_avg_speed': np.random.normal(45, 15, n_samples),
        'segment_length_km': np.random.uniform(0.5, 10, n_samples),
        'num_traffic_lights': np.random.randint(0, 10, n_samples),
        'is_highway': np.random.binomial(1, 0.3, n_samples),
        'incident_nearby': np.random.binomial(1, 0.05, n_samples)
    }

    df = pd.DataFrame(data)

    # Generate delay labels based on conditions
    delays = []
    for idx, row in df.iterrows():
        base_delay = 2  # Base 2 minutes

        # Rush hour impact
        hour = row['timestamp'].hour
        if (7 <= hour <= 9) or (17 <= hour <= 19):
            base_delay += np.random.uniform(5, 15)

        # Weather impact
        if row['rain_mm'] > 5:
            base_delay += np.random.uniform(3, 10)

        # Incident impact
        if row['incident_nearby']:
            base_delay += np.random.uniform(10, 30)

        # Segment characteristics
        if not row['is_highway']:
            base_delay += row['num_traffic_lights'] * 0.5

        # Add noise
        base_delay += np.random.normal(0, 2)

        delays.append(max(0, base_delay))

    return df, np.array(delays)


if __name__ == "__main__":
    # Train and save model
    print("Training Traffic ETA Model...")
    X, y = generate_synthetic_traffic_data(3000)

    model = TrafficETAModel()
    metrics = model.train(X, y)
    print(f"Training completed: {metrics}")

    # Save model
    model.save("models/traffic_eta_model.pkl", "models/traffic_eta_scaler.pkl")
    print("Model saved successfully!")

    # Test prediction
    test_input = {
        'timestamp': '2024-11-11T08:30:00',
        'temperature': 22, 'rain_mm': 8, 'visibility_km': 5,
        'historical_avg_speed': 30, 'segment_length_km': 5,
        'num_traffic_lights': 6, 'is_highway': 0, 'incident_nearby': 1
    }
    delay, confidence, explanation = model.predict(test_input)
    print(f"\nTest Prediction: {delay:.2f} minutes delay (confidence: {confidence:.2f})")
    print(f"Explanation: {explanation}")
