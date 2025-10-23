# ML Model Serving API with MLOps

A production-ready machine learning model serving API with comprehensive MLOps capabilities including model versioning, monitoring, and automated training pipelines.

## Features

- **RESTful API** for model predictions with FastAPI  
- **Model Versioning** with support for multiple model formats  
- **Real-time Monitoring** with Prometheus and Grafana  
- **Data Drift Detection** and model performance tracking  
- **Automated Training Pipelines** with model registry  
- **Docker & Kubernetes** ready deployment  
- **CI/CD Integration** with GitHub Actions  

## Tech Stack

| Category | Technology |
|-----------|-------------|
| **API Framework** | FastAPI (async/await) |
| **Machine Learning** | Scikit-learn, TensorFlow, PyTorch |
| **Monitoring** | Prometheus, Grafana |
| **Database** | PostgreSQL (async) |
| **Storage** | Local filesystem, S3, or cloud storage |
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Kubernetes manifests |
| **CI/CD** | GitHub Actions |

## Architecture

The system follows a microservices architecture:

1. **API Service:** Handles prediction requests and model management  
2. **Model Registry:** Manages model versions and metadata  
3. **Monitoring Service:** Tracks predictions, errors, and data drift  
4. **Training Pipeline:** Automated model training and evaluation  
5. **Database:** Stores model metadata and prediction history  

## Quick Start

### Prerequisites

- Python 3.9+  
- Docker and Docker Compose  
- PostgreSQL (for production)  

### Local Development

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/ml-model-serving.git
   cd ml-model-serving
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements/dev.txt
   ```

4. **Run with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

5. **Access the services**:
   - API: http://localhost:8000  
   - API Docs: http://localhost:8000/docs  
   - Grafana: http://localhost:3000 (admin/admin)  
   - Prometheus: http://localhost:9090  

## API Usage

**Make a prediction**:
```bash
curl -X POST "http://localhost:8000/api/v1/predict" -H "Content-Type: application/json" -d '{
  "model_version": "v1",
  "features": [[5.1, 3.5, 1.4, 0.2]]
}'
```

**List available models**:
```bash
curl "http://localhost:8000/api/v1/models"
```

**Get model statistics**:
```bash
curl "http://localhost:8000/api/v1/monitoring/models/v1/stats"
```

## Model Formats Supported

- Pickle (.pkl)  
- Joblib (.joblib)  
- TensorFlow/Keras (.h5)  
- ONNX (.onnx)  
- PyTorch (.pt) - via custom loading  

## Monitoring and Metrics

The API exposes Prometheus metrics at `/metrics`:

- `model_predictions_total`: Total predictions count  
- `model_prediction_latency_seconds`: Prediction latency histogram  
- `model_prediction_errors_total`: Error counts by type  
- `model_throughput_predictions_per_second`: Real-time throughput  

## Training Pipeline

The training pipeline includes:

1. Data validation and preprocessing  
2. Model training with hyperparameter tuning  
3. Model evaluation and validation  
4. Model registration and versioning  
5. Automated testing and deployment  

**Run the training pipeline**:
```bash
python -m training_pipeline.pipeline
```

## Deployment

### Kubernetes

**Deploy to Kubernetes**:
```bash
kubectl apply -f kubernetes/
```

## Cloud Deployment

The application can be deployed to:

- AWS ECS/EKS  
- Google Cloud Run/GKE  
- Azure Container Instances/AKS  
- Heroku with Docker  

## CI/CD Pipeline

GitHub Actions workflows included:

- CI: Run tests on pull requests  
- CD: Deploy to staging/production  
- Training: Automated model training pipeline  
- Monitoring: Data drift detection and alerts  

## Contributing

1. Fork the repository  
2. Create a feature branch  
3. Make your changes  
4. Add tests and ensure they pass  
5. Submit a pull request  

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
