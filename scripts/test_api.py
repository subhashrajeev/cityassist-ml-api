"""
Test script for CityAssist DS API
Run this to verify all endpoints are working correctly
"""
import requests
import json
from pathlib import Path

API_BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)

    response = requests.get(f"{API_BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    print("✓ Health check passed!")

def test_aqi_alert():
    """Test AQI alert prediction"""
    print("\n" + "="*60)
    print("Testing AQI Alert Prediction")
    print("="*60)

    payload = {
        "zone_id": "Zone-A",
        "pm25": 180,
        "pm10": 250,
        "no2": 45,
        "so2": 25,
        "co": 1.5,
        "o3": 50,
        "pm25_change_pct": 60,
        "temperature": 28,
        "humidity": 65,
        "wind_speed": 8,
        "user_age": 35,
        "has_health_condition": 1
    }

    response = requests.post(
        f"{API_BASE_URL}/api/v1/predict/aqi-alert",
        json=payload
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "alert_level" in response.json()["prediction"]
    print("✓ AQI alert prediction passed!")

def test_route_eta():
    """Test route ETA prediction"""
    print("\n" + "="*60)
    print("Testing Route ETA Prediction")
    print("="*60)

    payload = {
        "origin": "Location A",
        "destination": "Location B",
        "segments": [
            {
                "segment_id": "seg-1",
                "timestamp": "2024-11-11T08:30:00",
                "temperature": 22,
                "rain_mm": 8,
                "visibility_km": 5,
                "historical_avg_speed": 30,
                "segment_length_km": 5,
                "num_traffic_lights": 6,
                "is_highway": 0,
                "incident_nearby": 1
            },
            {
                "segment_id": "seg-2",
                "timestamp": "2024-11-11T08:30:00",
                "temperature": 22,
                "rain_mm": 2,
                "visibility_km": 10,
                "historical_avg_speed": 50,
                "segment_length_km": 3,
                "num_traffic_lights": 2,
                "is_highway": 1,
                "incident_nearby": 0
            }
        ]
    }

    response = requests.post(
        f"{API_BASE_URL}/api/v1/predict/route-eta",
        json=payload
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert "eta" in response.json()
    assert "total_eta_minutes" in response.json()["eta"]
    print("✓ Route ETA prediction passed!")

def test_outage_eta():
    """Test outage ETA prediction"""
    print("\n" + "="*60)
    print("Testing Outage ETA Prediction")
    print("="*60)

    payload = {
        "outage_id": "outage-123",
        "timestamp": "2024-11-11T14:30:00",
        "cause": "storm_damage",
        "zone": "zone_C",
        "temperature": 15,
        "wind_speed": 55,
        "affected_customers": 2500,
        "historical_mean_restore_hours": 5.5
    }

    response = requests.post(
        f"{API_BASE_URL}/api/v1/predict/outage-eta",
        json=payload
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "restore_hours" in response.json()["prediction"]
    print("✓ Outage ETA prediction passed!")

def test_image_classification():
    """Test image classification"""
    print("\n" + "="*60)
    print("Testing Image Classification")
    print("="*60)

    # Create a simple test image
    from PIL import Image
    import io

    # Create a red image (pothole-like)
    img = Image.new('RGB', (224, 224), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)

    files = {'file': ('test.png', img_bytes, 'image/png')}

    response = requests.post(
        f"{API_BASE_URL}/api/v1/classify/report-image",
        files=files
    )

    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "label" in response.json()["prediction"]
    print("✓ Image classification passed!")

def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("CityAssist DS API - Comprehensive Test Suite")
    print("="*60)
    print(f"\nTesting API at: {API_BASE_URL}")
    print("Make sure the API is running before running tests!")
    print("\nStarting tests...\n")

    try:
        test_health()
        test_aqi_alert()
        test_route_eta()
        test_outage_eta()
        test_image_classification()

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe API is working correctly and ready for deployment!")

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print(f"Make sure the API is running at {API_BASE_URL}")
        print("Run: python api/main.py")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {str(e)}")

    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}")

if __name__ == "__main__":
    run_all_tests()
