"""
Outage ETA Estimation Model
Predicts time-to-restore for utility outages
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from typing import Dict, Tuple, Any

class OutageETAModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.cause_encoder = LabelEncoder()
        self.zone_encoder = LabelEncoder()
        self.feature_names = [
            'cause_encoded', 'zone_encoded', 'hour', 'day_of_week', 'is_weekend',
            'temperature', 'wind_speed', 'is_storm',
            'historical_mean_restore_hours', 'affected_customers',
            'severity_score'
        ]

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for outage ETA prediction"""
        df = data.copy()

        # Temporal features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Weather impact
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['is_storm'] = ((df['wind_speed'] > 40) |
                              (df['temperature'] < 0) |
                              (df['temperature'] > 40)).astype(int)

        # Encode categorical variables
        if 'cause' in df.columns:
            if not hasattr(self.cause_encoder, 'classes_'):
                self.cause_encoder.fit(df['cause'])
            df['cause_encoded'] = self.cause_encoder.transform(df['cause'])

        if 'zone' in df.columns:
            if not hasattr(self.zone_encoder, 'classes_'):
                self.zone_encoder.fit(df['zone'])
            df['zone_encoded'] = self.zone_encoder.transform(df['zone'])

        # Severity score
        if 'affected_customers' in df.columns:
            df['severity_score'] = np.log1p(df['affected_customers'])

        # Fill missing values (only for numeric columns)
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

        return df

    def train(self, X: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        """Train the outage ETA prediction model"""
        # Feature engineering
        X_features = self.create_features(X)

        # Use only available features
        available_features = [f for f in self.feature_names if f in X_features.columns]
        X_train = X_features[available_features]

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=10,
            random_state=42
        )
        self.model.fit(X_scaled, y)

        # Training metrics
        train_score = self.model.score(X_scaled, y)
        predictions = self.model.predict(X_scaled)
        mae = np.mean(np.abs(predictions - y))
        rmse = np.sqrt(np.mean((predictions - y) ** 2))

        return {
            "train_r2_score": train_score,
            "train_mae_hours": mae,
            "train_rmse_hours": rmse,
            "n_features": len(available_features),
            "features": available_features
        }

    def predict(self, features: Dict[str, Any]) -> Tuple[float, float, float, Dict[str, Any]]:
        """
        Predict outage restoration time
        Returns: (restore_hours, lower_bound, upper_bound, explanation)
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
        restore_hours = float(self.model.predict(X_scaled)[0])

        # Calculate prediction intervals (simple approach using quantiles)
        # In production, use quantile regression or conformal prediction
        lower_bound = restore_hours * 0.7
        upper_bound = restore_hours * 1.3

        # Generate explanation
        explanation = self._generate_explanation(features, restore_hours)

        return restore_hours, lower_bound, upper_bound, explanation

    def _generate_explanation(self, features: Dict, restore_hours: float) -> Dict[str, Any]:
        """Generate human-readable explanation for prediction"""
        explanation = {
            "reason": "",
            "key_factors": [],
            "estimated_restoration": ""
        }

        # Rule-based reasons
        cause = features.get('cause', 'unknown')
        affected = features.get('affected_customers', 0)
        is_storm = features.get('is_storm', 0)
        wind_speed = features.get('wind_speed', 0)

        reasons = []

        # Cause-based reasoning
        if cause == 'equipment_failure':
            reasons.append("equipment failure requires replacement")
        elif cause == 'tree_fall':
            reasons.append("clearing fallen trees takes time")
        elif cause == 'storm_damage':
            reasons.append("storm damage requires extensive repairs")
        elif cause == 'planned_maintenance':
            reasons.append("scheduled maintenance window")

        # Scale impact
        if affected > 1000:
            reasons.append(f"large outage affecting {affected} customers")
        if is_storm:
            reasons.append(f"adverse weather conditions (wind: {wind_speed:.0f} km/h)")

        if reasons:
            explanation["reason"] = f"Extended restoration time due to: {', '.join(reasons)}"
        else:
            explanation["reason"] = "Standard restoration procedures apply"

        # Estimated time message
        hours = int(restore_hours)
        minutes = int((restore_hours - hours) * 60)
        explanation["estimated_restoration"] = f"Approximately {hours}h {minutes}m"

        # Feature importance
        if self.model:
            feature_importance = self.model.feature_importances_
            available_features = [f for f in self.feature_names if f in features]
            for i, feat in enumerate(available_features[:3]):
                if i < len(feature_importance):
                    explanation["key_factors"].append({
                        "feature": feat,
                        "importance": float(feature_importance[i])
                    })

        return explanation

    def save(self, model_path: str, scaler_path: str):
        """Save model, scaler, and encoders"""
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'cause_encoder': self.cause_encoder,
            'zone_encoder': self.zone_encoder
        }, model_path)

    def load(self, model_path: str, scaler_path: str = None):
        """Load model, scaler, and encoders"""
        artifacts = joblib.load(model_path)
        self.model = artifacts['model']
        self.scaler = artifacts['scaler']
        self.cause_encoder = artifacts['cause_encoder']
        self.zone_encoder = artifacts['zone_encoder']


def generate_synthetic_outage_data(n_samples: int = 1500) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate synthetic outage data for training"""
    np.random.seed(42)

    causes = ['equipment_failure', 'tree_fall', 'storm_damage', 'vehicle_accident',
              'planned_maintenance', 'overload', 'unknown']
    zones = ['zone_A', 'zone_B', 'zone_C', 'zone_D', 'zone_E']

    data = {
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='3H'),
        'cause': np.random.choice(causes, n_samples),
        'zone': np.random.choice(zones, n_samples),
        'temperature': np.random.normal(20, 10, n_samples),
        'wind_speed': np.random.exponential(20, n_samples),
        'affected_customers': np.random.lognormal(5, 2, n_samples),
        'historical_mean_restore_hours': np.random.uniform(0.5, 8, n_samples)
    }

    df = pd.DataFrame(data)

    # Generate restoration time labels based on conditions
    restore_times = []
    for idx, row in df.iterrows():
        base_time = 2  # Base 2 hours

        # Cause impact
        cause_multipliers = {
            'equipment_failure': 3,
            'tree_fall': 2.5,
            'storm_damage': 4,
            'vehicle_accident': 2,
            'planned_maintenance': 1,
            'overload': 1.5,
            'unknown': 2
        }
        base_time *= cause_multipliers.get(row['cause'], 2)

        # Weather impact
        if row['wind_speed'] > 40:
            base_time *= 1.5

        # Scale impact
        if row['affected_customers'] > 1000:
            base_time *= 1.3

        # Historical mean influence
        base_time = 0.7 * base_time + 0.3 * row['historical_mean_restore_hours']

        # Add noise
        base_time += np.random.normal(0, 0.5)

        restore_times.append(max(0.5, base_time))

    return df, np.array(restore_times)


if __name__ == "__main__":
    # Train and save model
    print("Training Outage ETA Model...")
    X, y = generate_synthetic_outage_data(2000)

    model = OutageETAModel()
    metrics = model.train(X, y)
    print(f"Training completed: {metrics}")

    # Save model
    model.save("models/outage_eta_model.pkl", "models/outage_eta_scaler.pkl")
    print("Model saved successfully!")

    # Test prediction
    test_input = {
        'timestamp': '2024-11-11T14:30:00',
        'cause': 'storm_damage',
        'zone': 'zone_C',
        'temperature': 15,
        'wind_speed': 55,
        'affected_customers': 2500,
        'historical_mean_restore_hours': 5.5
    }
    hours, lower, upper, explanation = model.predict(test_input)
    print(f"\nTest Prediction: {hours:.2f} hours ({lower:.2f} - {upper:.2f})")
    print(f"Explanation: {explanation}")
