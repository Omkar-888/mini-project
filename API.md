# Face Recognition Attendance API Documentation

## 🚀 Overview

The **Face Recognition Attendance API** is a FastAPI-based REST service that provides programmatic access to the face recognition attendance system. It's built on top of the `AttendanceSystem` class and offers endpoints compatible with the `main.py` structure while providing full functionality.

## 📋 Table of Contents

- [🛠️ Setup & Installation](#setup--installation)
- [🚀 Starting the Server](#starting-the-server)
- [🔗 API Endpoints](#api-endpoints)
- [📝 Request/Response Models](#requestresponse-models)
- [💡 Usage Examples](#usage-examples)
- [🔧 Configuration](#configuration)
- [🚨 Error Handling](#error-handling)
- [🧪 Testing](#testing)

## 🛠️ Setup & Installation

### Prerequisites

```bash
# Install required dependencies
pip install fastapi uvicorn python-multipart

# Ensure attendance system is ready
python enrolment.py  # Create enrollment database first
```

### File Structure

```
mini-project/
├── microservices.py         # FastAPI server
├── attendance_system.py     # Core attendance system
├── enrolment.py            # Enrollment system (required)
├── database/
│   ├── enrolment.pkl       # Face database (required)
│   └── attendance.pkl      # Attendance records (auto-created)
└── mediapipe_models/       # TFLite models (auto-downloaded)
```

## 🚀 Starting the Server

### Method 1: Direct Execution

```bash
python microservices.py
```

### Method 2: Using Uvicorn

```bash
uvicorn microservices:app --host 0.0.0.0 --port 8000 --reload
```

### Server Information

- **Default URL**: `http://localhost:8000`
- **Interactive API Docs**: `http://localhost:8000/docs` (Swagger UI)
- **Alternative Docs**: `http://localhost:8000/redoc` (ReDoc)
- **OpenAPI Schema**: `http://localhost:8000/openapi.json`

## 🔗 API Endpoints

### 🏠 System Status

#### `GET /`

**Root endpoint - System status and information**

```bash
curl -X GET "http://localhost:8000/"
```

**Response:**

```json
{
  "status": "success",
  "message": "Face Recognition Attendance API is running",
  "system_ready": true,
  "enrolled_persons": 5,
  "similarity_threshold": 0.7
}
```

#### `GET /health`

**Health check endpoint**

```bash
curl -X GET "http://localhost:8000/health"
```

**Response:**

```json
{
  "status": "healthy",
  "timestamp": "2025-08-15T14:30:15.123456",
  "system_ready": true
}
```

### 🎯 Face Recognition (Compatible with main.py)

#### `GET /attendance/recognise`

**Get latest recognition result**

```bash
curl -X GET "http://localhost:8000/attendance/recognise"
```

**Response:**

```json
{
  "user_id": "alice_smith",
  "confidence_score": 0.847
}
```

#### `POST /attendance/recognise`

**Upload images for recognition (Multipart mode)**

```bash
curl -X POST "http://localhost:8000/attendance/recognise" \
  -H "Content-Type: multipart/form-data" \
  -F "frames=@photo1.jpg" \
  -F "frames=@photo2.jpg" \
  -F "multipart_mode=true"
```

**Parameters:**

- `frames`: List of image files (multipart/form-data)
- `multipart_mode`: Boolean (default: true) - Consolidates all images as one person attempt

**Response:**

```json
{
  "status": "success",
  "images_processed": 2,
  "multipart_mode": true
}
```

#### `POST /attendance/recognise-single`

**Process each image individually (Legacy mode)**

```bash
curl -X POST "http://localhost:8000/attendance/recognise-single" \
  -H "Content-Type: multipart/form-data" \
  -F "frames=@photo1.jpg" \
  -F "frames=@photo2.jpg"
```

**Response:**

```json
{
  "status": "success",
  "images_processed": 2,
  "multipart_mode": false
}
```

### 📊 Detailed Results

#### `GET /attendance/batch-results`

**Get detailed batch processing results**

```bash
curl -X GET "http://localhost:8000/attendance/batch-results"
```

**Response:**

```json
{
  "total_images": 3,
  "total_faces": 3,
  "results": [
    {
      "user_id": "alice_smith",
      "confidence_score": 0.847
    }
  ],
  "processing_time_ms": 1234.5,
  "timestamp": "2025-08-15T14:30:15.123456"
}
```

#### `GET /attendance/voting-details`

**Get detailed voting information from multipart processing**

```bash
curl -X GET "http://localhost:8000/attendance/voting-details"
```

**Response:**

```json
{
  "total_images": 3,
  "total_faces": 3,
  "processing_time_ms": 1234.5,
  "final_result": {
    "user_id": "alice_smith",
    "confidence_score": 0.892
  },
  "timestamp": "2025-08-15T14:30:15.123456"
}
```

### 👥 System Information

#### `GET /attendance/enrolled-persons`

**Get list of all enrolled persons**

```bash
curl -X GET "http://localhost:8000/attendance/enrolled-persons"
```

**Response:**

```json
{
  "total_persons": 5,
  "persons": [
    "alice_smith",
    "bob_jones",
    "charlie_brown",
    "diana_wilson",
    "eve_johnson"
  ],
  "database_path": "database/enrolment.pkl",
  "similarity_threshold": 0.7
}
```

### ⚙️ Configuration

#### `POST /attendance/config`

**Update system configuration**

```bash
curl -X POST "http://localhost:8000/attendance/config" \
  -H "Content-Type: application/json" \
  -d '{"similarity_threshold": 0.8}'
```

**Request Body:**

```json
{
  "similarity_threshold": 0.8
}
```

## 📝 Request/Response Models

### PredictionResponse

```python
{
  "user_id": str,           # Identified person name or "unknown"
  "confidence_score": float # Recognition confidence (0.0-1.0)
}
```

### BatchPredictionResponse

```python
{
  "total_images": int,      # Number of images processed
  "total_faces": int,       # Number of faces detected
  "results": [              # List of PredictionResponse objects
    {
      "user_id": str,
      "confidence_score": float
    }
  ],
  "processing_time_ms": float,  # Processing time in milliseconds
  "timestamp": str              # ISO timestamp
}
```

### Error Response

```python
{
  "status": "error",
  "message": str,           # Error message
  "detail": str,           # Detailed error information
  "timestamp": str         # ISO timestamp
}
```

## 💡 Usage Examples

### Python Client Example

```python
import requests
import json

# API base URL
API_URL = "http://localhost:8000"

class AttendanceAPIClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url

    def check_health(self):
        """Check API health status"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

    def upload_images_multipart(self, image_paths: list, multipart_mode: bool = True):
        """Upload multiple images for recognition"""
        files = []
        for path in image_paths:
            files.append(('frames', open(path, 'rb')))

        try:
            response = requests.post(
                f"{self.base_url}/attendance/recognise",
                files=files,
                data={'multipart_mode': multipart_mode}
            )
            return response.json()
        finally:
            # Close file handles
            for _, file_handle in files:
                file_handle.close()

    def get_latest_result(self):
        """Get latest recognition result"""
        response = requests.get(f"{self.base_url}/attendance/recognise")
        return response.json()

    def get_batch_results(self):
        """Get detailed batch results"""
        response = requests.get(f"{self.base_url}/attendance/batch-results")
        return response.json()

    def get_enrolled_persons(self):
        """Get list of enrolled persons"""
        response = requests.get(f"{self.base_url}/attendance/enrolled-persons")
        return response.json()

    def update_threshold(self, threshold: float):
        """Update similarity threshold"""
        response = requests.post(
            f"{self.base_url}/attendance/config",
            json={"similarity_threshold": threshold}
        )
        return response.json()

# Usage example
if __name__ == "__main__":
    client = AttendanceAPIClient()

    # Check system health
    health = client.check_health()
    print(f"System status: {health['status']}")

    # Upload images for recognition
    image_paths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"]
    result = client.upload_images_multipart(image_paths, multipart_mode=True)
    print(f"Upload result: {result}")

    # Get recognition result
    recognition = client.get_latest_result()
    print(f"Identified: {recognition['user_id']} (confidence: {recognition['confidence_score']:.3f})")

    # Get detailed results
    batch_results = client.get_batch_results()
    print(f"Processed {batch_results['total_images']} images in {batch_results['processing_time_ms']:.1f}ms")
```

### JavaScript/Node.js Example

```javascript
const axios = require("axios");
const FormData = require("form-data");
const fs = require("fs");

class AttendanceAPIClient {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
  }

  async checkHealth() {
    const response = await axios.get(`${this.baseUrl}/health`);
    return response.data;
  }

  async uploadImagesMultipart(imagePaths, multipartMode = true) {
    const formData = new FormData();

    // Add image files
    for (const path of imagePaths) {
      formData.append("frames", fs.createReadStream(path));
    }

    // Add multipart mode parameter
    formData.append("multipart_mode", multipartMode);

    const response = await axios.post(
      `${this.baseUrl}/attendance/recognise`,
      formData,
      {
        headers: {
          ...formData.getHeaders(),
        },
      }
    );

    return response.data;
  }

  async getLatestResult() {
    const response = await axios.get(`${this.baseUrl}/attendance/recognise`);
    return response.data;
  }

  async getBatchResults() {
    const response = await axios.get(
      `${this.baseUrl}/attendance/batch-results`
    );
    return response.data;
  }

  async getEnrolledPersons() {
    const response = await axios.get(
      `${this.baseUrl}/attendance/enrolled-persons`
    );
    return response.data;
  }

  async updateThreshold(threshold) {
    const response = await axios.post(`${this.baseUrl}/attendance/config`, {
      similarity_threshold: threshold,
    });
    return response.data;
  }
}

// Usage example
(async () => {
  const client = new AttendanceAPIClient();

  try {
    // Check system health
    const health = await client.checkHealth();
    console.log(`System status: ${health.status}`);

    // Upload images
    const imagePaths = ["photo1.jpg", "photo2.jpg", "photo3.jpg"];
    const result = await client.uploadImagesMultipart(imagePaths, true);
    console.log(`Upload result:`, result);

    // Get recognition result
    const recognition = await client.getLatestResult();
    console.log(
      `Identified: ${
        recognition.user_id
      } (confidence: ${recognition.confidence_score.toFixed(3)})`
    );
  } catch (error) {
    console.error("API Error:", error.response?.data || error.message);
  }
})();
```

### cURL Examples

#### Basic Recognition Workflow

```bash
# 1. Check system health
curl -X GET "http://localhost:8000/health"

# 2. Upload single image
curl -X POST "http://localhost:8000/attendance/recognise" \
  -H "Content-Type: multipart/form-data" \
  -F "frames=@photo.jpg" \
  -F "multipart_mode=true"

# 3. Get recognition result
curl -X GET "http://localhost:8000/attendance/recognise"

# 4. Get detailed batch results
curl -X GET "http://localhost:8000/attendance/batch-results"

# 5. Get voting details (for multipart processing)
curl -X GET "http://localhost:8000/attendance/voting-details"
```

#### Multiple Images (Multipart Mode)

```bash
# Upload multiple images as one person attempt
curl -X POST "http://localhost:8000/attendance/recognise" \
  -H "Content-Type: multipart/form-data" \
  -F "frames=@person_photo1.jpg" \
  -F "frames=@person_photo2.jpg" \
  -F "frames=@person_photo3.jpg" \
  -F "multipart_mode=true"
```

#### Configuration Management

````bash
# Get enrolled persons
curl -X GET "http://localhost:8000/attendance/enrolled-persons"

## 🔧 Configuration

### Environment Variables

```bash
# Optional environment variables
export ATTENDANCE_DB_PATH="database/enrolment.pkl"
export ATTENDANCE_THRESHOLD="0.7"
export API_HOST="0.0.0.0"
export API_PORT="8000"
````

### Server Configuration

```python
# In microservices.py, modify these settings:

# Similarity threshold (0.0 - 1.0)
SIMILARITY_THRESHOLD = 0.7

# Server settings
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# Enable auto-reload for development
RELOAD_MODE = True
```

### CORS Configuration (if needed)

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🚨 Error Handling

### Common HTTP Status Codes

| Status Code | Description           | Typical Cause                           |
| ----------- | --------------------- | --------------------------------------- |
| **200**     | Success               | Request processed successfully          |
| **400**     | Bad Request           | Invalid parameters or malformed request |
| **422**     | Validation Error      | Invalid input data format               |
| **500**     | Internal Server Error | Server-side processing error            |
| **503**     | Service Unavailable   | Attendance system not initialized       |

### Error Response Format

```json
{
  "status": "error",
  "message": "Descriptive error message",
  "detail": "Detailed error information",
  "timestamp": "2025-08-15T14:30:15.123456"
}
```

### Common Error Scenarios

#### 1. System Not Initialized

```json
{
  "detail": "Attendance system not initialized"
}
```

**Solution**: Ensure `database/enrolment.pkl` exists and contains enrolled persons

#### 2. No Images Uploaded

```json
{
  "detail": "No frames provided"
}
```

**Solution**: Include at least one image file in the request

#### 3. Invalid Image Format

```json
{
  "detail": "Error processing uploaded file"
}
```

**Solution**: Use supported formats (JPG, PNG, BMP, TIFF, WebP)

#### 4. Invalid Threshold Value

```json
{
  "detail": "Similarity threshold must be between 0.0 and 1.0"
}
```

**Solution**: Provide threshold value in valid range

### Error Handling in Client Code

```python
import requests

def safe_api_call(url, **kwargs):
    try:
        response = requests.get(url, **kwargs)
        response.raise_for_status()  # Raises HTTPError for bad status codes
        return response.json()
    except requests.exceptions.HTTPError as e:
        error_data = response.json() if response.content else {}
        print(f"HTTP Error {response.status_code}: {error_data.get('detail', str(e))}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        return None

# Usage
result = safe_api_call("http://localhost:8000/attendance/recognise")
if result:
    print(f"Recognition result: {result}")
```

## 🧪 Testing

### Unit Testing with pytest

```python
# test_api.py
import pytest
from fastapi.testclient import TestClient
from microservices import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "timestamp" in data

def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "system_ready" in data

def test_get_recognition_result():
    response = client.get("/attendance/recognise")
    assert response.status_code == 200
    data = response.json()
    assert "user_id" in data
    assert "confidence_score" in data

def test_upload_images():
    # Mock image file for testing
    test_files = [
        ("frames", ("test1.jpg", b"fake image data", "image/jpeg")),
        ("frames", ("test2.jpg", b"fake image data", "image/jpeg"))
    ]

    response = client.post(
        "/attendance/recognise",
        files=test_files,
        data={"multipart_mode": True}
    )

    # Note: This will fail without actual face data, but tests the endpoint structure
    assert response.status_code in [200, 500]  # 500 expected with fake data

def test_get_enrolled_persons():
    response = client.get("/attendance/enrolled-persons")
    assert response.status_code in [200, 503]  # 503 if system not initialized

def test_config_update():
    response = client.post(
        "/attendance/config",
        json={"similarity_threshold": 0.8}
    )
    assert response.status_code in [200, 503]  # 503 if system not initialized

def test_invalid_threshold():
    response = client.post(
        "/attendance/config",
        json={"similarity_threshold": 1.5}  # Invalid threshold
    )
    assert response.status_code in [400, 503]

# Run tests
# pytest test_api.py -v
```

### Integration Testing

```python
# integration_test.py
import requests
import time
import os

API_URL = "http://localhost:8000"

def test_full_workflow():
    """Test complete API workflow"""

    # 1. Check system health
    print("1. Testing health check...")
    response = requests.get(f"{API_URL}/health")
    assert response.status_code == 200
    print("✅ Health check passed")

    # 2. Get enrolled persons
    print("2. Testing enrolled persons endpoint...")
    response = requests.get(f"{API_URL}/attendance/enrolled-persons")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Found {data['total_persons']} enrolled persons")
    else:
        print("⚠️ No enrolled persons found")

    # 3. Upload test images (if available)
    test_images = ["test1.jpg", "test2.jpg"]
    available_images = [img for img in test_images if os.path.exists(img)]

    if available_images:
        print(f"3. Testing image upload with {len(available_images)} images...")
        files = [('frames', open(img, 'rb')) for img in available_images]

        try:
            response = requests.post(
                f"{API_URL}/attendance/recognise",
                files=files,
                data={'multipart_mode': True}
            )
            print(f"Upload status: {response.status_code}")

            # 4. Get recognition result
            time.sleep(1)  # Allow processing time
            response = requests.get(f"{API_URL}/attendance/recognise")
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Recognition result: {result['user_id']} (confidence: {result['confidence_score']:.3f})")

        finally:
            for _, file_handle in files:
                file_handle.close()
    else:
        print("⚠️ No test images available, skipping upload test")

    print("🎉 Integration test completed")

if __name__ == "__main__":
    test_full_workflow()
```

### Load Testing with Locust

```python
# locustfile.py
from locust import HttpUser, task, between

class AttendanceAPIUser(HttpUser):
    wait_time = between(1, 3)

    def on_start(self):
        """Called when user starts"""
        # Check if system is ready
        response = self.client.get("/health")
        if response.status_code != 200:
            print("System not ready!")

    @task(3)
    def get_health_check(self):
        """Health check endpoint (frequent)"""
        self.client.get("/health")

    @task(2)
    def get_recognition_result(self):
        """Get latest recognition result"""
        self.client.get("/attendance/recognise")

    @task(1)
    def get_enrolled_persons(self):
        """Get enrolled persons (less frequent)"""
        self.client.get("/attendance/enrolled-persons")

    @task(1)
    def get_batch_results(self):
        """Get batch results"""
        self.client.get("/attendance/batch-results")

# Run load test:
# locust -f locustfile.py --host=http://localhost:8000
```

## 🚀 Production Deployment

### Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "microservices:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: "3.8"
services:
  attendance-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./database:/app/database
      - ./mediapipe_models:/app/mediapipe_models
    environment:
      - PYTHONPATH=/app
    restart: unless-stopped
```

### Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/attendance-api
server {
    listen 80;
    server_name your-domain.com;

    client_max_body_size 50M;  # Allow large image uploads

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeout settings for image processing
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

### Systemd Service

```ini
# /etc/systemd/system/attendance-api.service
[Unit]
Description=Face Recognition Attendance API
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/your/app
Environment=PATH=/path/to/your/venv/bin
ExecStart=/path/to/your/venv/bin/uvicorn microservices:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable attendance-api
sudo systemctl start attendance-api
sudo systemctl status attendance-api
```

## 📚 API Reference Summary

### Core Endpoints

| Method | Endpoint                       | Description                     |
| ------ | ------------------------------ | ------------------------------- |
| `GET`  | `/`                            | System status and information   |
| `GET`  | `/health`                      | Health check                    |
| `GET`  | `/attendance/recognise`        | Get latest recognition result   |
| `POST` | `/attendance/recognise`        | Upload images (multipart mode)  |
| `POST` | `/attendance/recognise-single` | Upload images (individual mode) |
| `GET`  | `/attendance/batch-results`    | Get detailed batch results      |
| `GET`  | `/attendance/voting-details`   | Get voting information          |
| `GET`  | `/attendance/enrolled-persons` | List enrolled persons           |
| `POST` | `/attendance/config`           | Update configuration            |

### Quick Start Commands

```bash
# 1. Start server
python microservices.py

# 2. Check health
curl http://localhost:8000/health

# 3. Upload image
curl -X POST http://localhost:8000/attendance/recognise \
  -F "frames=@photo.jpg" -F "multipart_mode=true"

# 4. Get result
curl http://localhost:8000/attendance/recognise
```

**🎉 Your Face Recognition Attendance API is ready for integration!**
