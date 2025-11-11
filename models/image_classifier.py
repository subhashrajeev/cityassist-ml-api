"""
Image Classification Model for Civic Reports
Classifies citizen-reported images (pothole, garbage, tree fall, etc.)
"""
import numpy as np
from PIL import Image
import io
from typing import Dict, Tuple, Any
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class CivicImageClassifier:
    def __init__(self, image_size=(224, 224), num_classes=5):
        self.image_size = image_size
        self.num_classes = num_classes
        self.model = None
        self.class_labels = ['pothole', 'garbage', 'tree_fall', 'streetlight', 'other']

    def build_model(self) -> keras.Model:
        """Build MobileNetV2-based image classification model"""
        # Use MobileNetV2 as backbone for efficient inference
        base_model = MobileNetV2(
            input_shape=(*self.image_size, 3),
            include_top=False,
            weights='imagenet'
        )

        # Freeze base model layers
        base_model.trainable = False

        # Build model
        inputs = keras.Input(shape=(*self.image_size, 3))

        # Preprocessing
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)

        # Base model
        x = base_model(x, training=False)

        # Custom head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)

        model = keras.Model(inputs, outputs)

        return model

    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = 10) -> Dict[str, Any]:
        """Train the image classification model"""
        # Build model
        self.model = self.build_model()

        # Compile
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )

        # Train
        history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            validation_data=(X_val, y_val) if X_val is not None else None,
            epochs=epochs,
            verbose=1
        )

        # Get metrics
        final_accuracy = history.history['accuracy'][-1]
        val_accuracy = history.history.get('val_accuracy', [0])[-1] if X_val is not None else 0

        return {
            "train_accuracy": final_accuracy,
            "val_accuracy": val_accuracy,
            "num_classes": self.num_classes,
            "image_size": self.image_size
        }

    def preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image bytes for prediction"""
        # Open image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize
        image = image.resize(self.image_size)

        # Convert to array and normalize
        img_array = np.array(image)

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image_bytes: bytes) -> Tuple[str, float, str, Dict[str, Any]]:
        """
        Predict image class
        Returns: (label, confidence, priority, explanation)
        """
        # Preprocess
        img_array = self.preprocess_image(image_bytes)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Get top prediction
        top_idx = np.argmax(predictions)
        confidence = float(predictions[top_idx])
        label = self.class_labels[top_idx]

        # Determine priority
        priority = self._determine_priority(label, confidence)

        # Generate explanation
        explanation = self._generate_explanation(label, confidence, predictions)

        return label, confidence, priority, explanation

    def predict_top_k(self, image_bytes: bytes, k: int = 3) -> Dict[str, Any]:
        """Predict top-k classes with probabilities"""
        # Preprocess
        img_array = self.preprocess_image(image_bytes)

        # Predict
        predictions = self.model.predict(img_array, verbose=0)[0]

        # Get top-k predictions
        top_k_idx = np.argsort(predictions)[-k:][::-1]

        results = []
        for idx in top_k_idx:
            results.append({
                "label": self.class_labels[idx],
                "confidence": float(predictions[idx])
            })

        return {
            "top_predictions": results,
            "all_probabilities": {
                self.class_labels[i]: float(predictions[i])
                for i in range(len(self.class_labels))
            }
        }

    def _determine_priority(self, label: str, confidence: float) -> str:
        """Determine priority level based on classification"""
        # High confidence thresholds
        if confidence < 0.5:
            return "low"  # Uncertain, needs human review

        # Priority mapping
        priority_map = {
            'pothole': 'high',      # Safety hazard
            'garbage': 'medium',    # Health/aesthetic issue
            'tree_fall': 'high',    # Safety hazard, blocks roads
            'streetlight': 'medium', # Utility issue
            'other': 'low'          # Needs review
        }

        return priority_map.get(label, 'low')

    def _generate_explanation(self, label: str, confidence: float,
                              all_predictions: np.ndarray) -> Dict[str, Any]:
        """Generate explanation for classification"""
        explanation = {
            "primary_class": label,
            "confidence_level": "high" if confidence > 0.8 else "medium" if confidence > 0.6 else "low",
            "requires_review": confidence < 0.7,
            "alternative_classes": []
        }

        # Add alternative classes if confidence is not very high
        if confidence < 0.9:
            sorted_idx = np.argsort(all_predictions)[-3:][::-1]
            for idx in sorted_idx[1:]:  # Skip the top prediction
                if all_predictions[idx] > 0.1:
                    explanation["alternative_classes"].append({
                        "class": self.class_labels[idx],
                        "probability": float(all_predictions[idx])
                    })

        return explanation

    def save(self, model_path: str):
        """Save model"""
        self.model.save(model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path: str):
        """Load model"""
        self.model = keras.models.load_model(model_path)
        print(f"Model loaded from {model_path}")


def generate_synthetic_image_data(n_samples: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic image data for training"""
    # Generate random images with different patterns
    X = []
    y = []

    for i in range(n_samples):
        # Create synthetic image with different characteristics per class
        class_label = i % 5

        # Create random image with class-specific patterns
        if class_label == 0:  # pothole - dark center
            img = np.random.rand(224, 224, 3) * 100
            img[80:140, 80:140] = np.random.rand(60, 60, 3) * 30  # Dark patch
        elif class_label == 1:  # garbage - mixed colors
            img = np.random.rand(224, 224, 3) * 200
        elif class_label == 2:  # tree fall - brown/green mix
            img = np.zeros((224, 224, 3))
            img[:, :, 0] = np.random.rand(224, 224) * 100  # Red channel
            img[:, :, 1] = np.random.rand(224, 224) * 150  # Green channel
        elif class_label == 3:  # streetlight - bright spot
            img = np.random.rand(224, 224, 3) * 50
            img[50:100, 100:120] = 255  # Bright area
        else:  # other
            img = np.random.rand(224, 224, 3) * 128

        X.append(img)
        y.append(class_label)

    return np.array(X).astype('float32'), np.array(y)


if __name__ == "__main__":
    # Train and save model
    print("Training Image Classification Model...")
    print("This may take several minutes...")

    # Generate synthetic data
    X_train, y_train = generate_synthetic_image_data(1000)
    X_val, y_val = generate_synthetic_image_data(200)

    model = CivicImageClassifier()
    metrics = model.train(X_train, y_train, X_val, y_val, epochs=5)
    print(f"Training completed: {metrics}")

    # Save model
    model.save("models/image_classifier.h5")
    print("Model saved successfully!")

    # Test with synthetic image
    test_img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    test_img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    label, confidence, priority, explanation = model.predict(img_bytes)
    print(f"\nTest Prediction: {label} (confidence: {confidence:.2f}, priority: {priority})")
    print(f"Explanation: {explanation}")
