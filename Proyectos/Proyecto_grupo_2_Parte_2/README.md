# MLOps Diabetes Readmission Prediction

A production-ready MLOps platform for predicting diabetes patient hospital readmission using Apache Airflow, MLflow, Docker, and Kubernetes. This project demonstrates end-to-end machine learning workflow automation including data ingestion, preprocessing, model training, experiment tracking, and deployment.

## Table of Contents

- [Video Explanation](#video-explanation)
- [Project Description](#project-description)
- [Quick Start Guide](#quick-start-guide)
- [Requirements](#requirements)
- [Architecture](#architecture)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [How to Run the Project](#how-to-run-the-project)
- [Deploy Inference Services](#step-7-deploy-inference-services-api-ui-monitoring)
- [Using the Inference API](#step-8-using-the-inference-api)
- [Using the Web UI](#step-9-using-the-web-ui)
- [Monitoring with Prometheus & Grafana](#step-10-monitoring-with-prometheus--grafana)
- [Load Testing with Locust](#step-11-load-testing-with-locust)

---

## Video Explanation

Watch the video demonstration of this MLOps platform:

[![Video Explanation](https://img.youtube.com/vi/IlCxL15DW5M/0.jpg)](https://youtu.be/IlCxL15DW5M)

**Video Link**: [https://youtu.be/IlCxL15DW5M](https://youtu.be/IlCxL15DW5M)

---

## Project Description

This MLOps platform automates the complete lifecycle of a diabetes patient readmission prediction model using an innovative **Cumulative Batch Training** approach. The system provides end-to-end ML workflow automation from data ingestion to production deployment, including model serving, monitoring, and load testing capabilities.

**Key Technologies**: Apache Airflow 3.1.0, MLflow 3.4.0, FastAPI, Streamlit, PostgreSQL 16, Redis 7.2, MinIO S3, Docker, Kubernetes (Minikube), Prometheus, Grafana, Locust, scikit-learn 1.4.2, Python 3.10

---

## Quick Start Guide

### Prerequisites
- Docker & Docker Compose installed
- Minikube installed and running
- kubectl installed
- 8GB+ RAM available

### Quick Start (5 Minutes)

1. **Start Minikube**:
   ```bash
   minikube start --driver=docker
   minikube addons enable metrics-server
   ```

2. **Deploy Databases**:
   ```bash
   cd k8s/komposedbfiles
   kubectl apply -f postgres-raw-data-deployment.yaml
   kubectl apply -f postgres-raw-data-service.yaml
   ```

3. **Start ML Services**:
   ```bash
   docker compose -f docker-compose.inference.yml up -d
   ```

4. **Train Model** (via Airflow UI at http://localhost:8001):
   - Enable `dag_00b_master_cumulative_pipeline`
   - Trigger the DAG
   - Wait for training to complete (~1-2 hours)

5. **Deploy Inference Services**:
   ```bash
   cd k8s/scripts
   ./deploy.sh --all
   ```

6. **Access Services**:
   - **API**: `http://<minikube-ip>:30080`
   - **UI**: `http://<minikube-ip>:30085`
   - **Prometheus**: `http://<minikube-ip>:30090`
   - **Grafana**: `http://<minikube-ip>:30300`
   - **Locust**: `http://<minikube-ip>:30189`

For detailed instructions, see [How to Run the Project](#how-to-run-the-project).

---

## Architecture

### Unified MLOps Platform Architecture

This diagram shows the complete architecture integrating Docker Compose services, Kubernetes databases, and the ML pipeline workflow.

```mermaid
graph TB
    subgraph "Data Source"
        DS[("fa:fa-cloud Google Drive<br/>Diabetes Dataset<br/>~101K records")]
    end

    subgraph "Docker Compose Environment"
        subgraph "Orchestration - Apache Airflow"
            AF_API["fa:fa-server Airflow API<br/>:8001"]
            AF_SCHED["fa:fa-calendar Scheduler"]
            AF_WORK["fa:fa-cog Worker<br/>Celery"]
            AF_DAG["fa:fa-code DAG Processor"]
            REDIS["fa:fa-database Redis<br/>:6379"]
            PG_AF[("fa:fa-database PostgreSQL<br/>Airflow Meta")]
        end

        subgraph "ML Training & Tracking - MLflow"
            MLFLOW["fa:fa-chart-line MLflow Server<br/>:8002"]
            MINIO["fa:fa-cubes MinIO S3<br/>:9000/:9001"]
            PG_ML[("fa:fa-database PostgreSQL<br/>MLflow Meta")]
        end

        subgraph "Data Storage - Docker"
            PG_RAW_D[("fa:fa-database postgres-raw<br/>:5433")]
            PG_CLEAN_D[("fa:fa-database postgres-clean<br/>:5434")]
        end
    end

    subgraph "Kubernetes - Minikube Cluster"
        K8S_NET["fa:fa-network-wired Minikube Network<br/>192.168.58.2"]

        subgraph "Database Deployments"
            K8S_RAW["fa:fa-server Raw Data Pod<br/>postgres:16<br/>NodePort: 31070"]
            K8S_CLEAN["fa:fa-server Clean Data Pod<br/>postgres:16<br/>NodePort: 30236"]
        end
    end

    subgraph "ML Pipeline Workflow - Cumulative Batch Training"
        P1["fa:fa-download DAG 01<br/>Download Data"]
        P2["fa:fa-cut DAG 02b<br/>Split & Batch<br/>70/15/15<br/>Batches: 15K each"]
        P3["fa:fa-table DAG 03b<br/>Store Batches<br/>+ Cumulative Views"]
        P4["fa:fa-layer-group Generate Configs<br/>Dynamic N batches"]
        P5["fa:fa-brain Parallel Training<br/>DAG 05b<br/>N×4 Models<br/>Cumulative 0→N"]
        P6["fa:fa-trophy DAG 06b<br/>Select Best<br/>F1 Score"]
        P7["fa:fa-rocket DAG 07<br/>Publish Production"]
    end

    %% Data Flow
    DS -->|Download| P1
    P1 --> P2
    P2 --> P3
    P3 -->|Batch Tables| PG_RAW_D
    P3 -->|Batch Tables| K8S_RAW
    P3 -->|Val/Test| PG_RAW_D
    P3 -->|Create Views| P4
    P4 -->|Config N Batches| P5

    PG_RAW_D -.->|Read Cumulative Views| P5
    K8S_RAW -.->|Read Cumulative Views| P5

    P5 -->|Log Experiments| MLFLOW
    P5 --> P6
    P6 -->|Evaluate| PG_RAW_D
    P6 --> P7
    P7 -->|Register Model| MLFLOW

    %% Airflow Orchestration
    AF_SCHED -.->|Schedule| P1
    AF_SCHED -.->|Schedule| P2
    AF_SCHED -.->|Schedule| P3
    AF_SCHED -.->|Schedule| P4
    AF_SCHED -.->|Schedule| P5
    AF_SCHED -.->|Schedule| P6
    AF_SCHED -.->|Schedule| P7
    AF_WORK -.->|Execute Tasks| P5
    AF_API -.->|Control| AF_SCHED
    AF_DAG -.->|Process| AF_SCHED
    REDIS -.->|Queue| AF_WORK
    PG_AF -.->|Metadata| AF_SCHED

    %% MLflow Components
    MLFLOW --> PG_ML
    MLFLOW --> MINIO

    %% Kubernetes Network
    K8S_NET --> K8S_RAW
    K8S_NET --> K8S_CLEAN
    AF_WORK -.->|Connect via<br/>Minikube Net| K8S_NET

    %% Styling with icons representation
    style DS fill:#e3f2fd,stroke:#1976d2,stroke-width:3px
    style AF_API fill:#bbdefb,stroke:#1565c0,stroke-width:2px
    style AF_SCHED fill:#bbdefb,stroke:#1565c0,stroke-width:2px
    style AF_WORK fill:#90caf9,stroke:#1565c0,stroke-width:2px
    style AF_DAG fill:#bbdefb,stroke:#1565c0,stroke-width:2px
    style REDIS fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style PG_AF fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style MLFLOW fill:#ffccbc,stroke:#e64a19,stroke-width:3px
    style MINIO fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style PG_ML fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style PG_RAW_D fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style PG_CLEAN_D fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style K8S_NET fill:#b2dfdb,stroke:#00695c,stroke-width:3px
    style K8S_RAW fill:#80cbc4,stroke:#00695c,stroke-width:2px
    style K8S_CLEAN fill:#80cbc4,stroke:#00695c,stroke-width:2px
    style P1 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style P2 fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style P3 fill:#fff59d,stroke:#f57f17,stroke-width:2px
    style P4 fill:#ffeb3b,stroke:#f57f17,stroke-width:2px
    style P5 fill:#ce93d8,stroke:#6a1b9a,stroke-width:3px
    style P6 fill:#ba68c8,stroke:#6a1b9a,stroke-width:2px
    style P7 fill:#81c784,stroke:#2e7d32,stroke-width:3px
```

### Cumulative Batch Pipeline Flow

This pipeline dynamically trains models on increasing data volumes to find the optimal training dataset size.

```mermaid
flowchart TD
    Start([Start Pipeline]) --> Download["fa:fa-download DAG 01<br/>Download Diabetes Dataset<br/>~101,767 records"]

    Download --> Batch["fa:fa-cut DAG 02b<br/>Create Batches<br/>Split: Train 70% / Val 15% / Test 15%<br/>Batch Size: 15,000 records"]

    Batch --> Store["fa:fa-table DAG 03b<br/>Store in PostgreSQL<br/>• 7 batch tables (batch_0 to batch_6)<br/>• Cumulative views (0→N)<br/>• Validation & Test tables"]

    Store --> Generate["fa:fa-layer-group Generate Configs<br/>Dynamic Configuration<br/>Based on actual batch count"]

    Generate --> Parallel["fa:fa-brain DAG 05b<br/>Parallel Cumulative Training<br/><br/>╔════════════════════════╗<br/>║ Cumulative 0: 15K      ║ → 4 models<br/>║ Cumulative 1: 30K      ║ → 4 models<br/>║ Cumulative 2: 45K      ║ → 4 models<br/>║ Cumulative 3: 60K      ║ → 4 models<br/>║ Cumulative 4: 75K      ║ → 4 models<br/>║ Cumulative 5: 90K      ║ → 4 models<br/>║ Cumulative 6: 102K     ║ → 4 models<br/>╚════════════════════════╝<br/><br/>Total: 7×4 = 28 models"]

    Parallel --> Models["fa:fa-flask Model Types per Batch<br/>1. Logistic Regression<br/>2. Decision Tree<br/>3. Random Forest<br/>4. Gradient Boosting"]

    Models --> Aggregate["fa:fa-table DAG 06b<br/>Aggregate Results<br/>Compare all 28 models"]

    Aggregate --> Select["fa:fa-trophy Select Best Model<br/>Rank by F1 Score<br/>Evaluate on full validation set"]

    Select --> Publish["fa:fa-rocket DAG 07<br/>Publish to Production<br/>MLflow Model Registry<br/>Stage: Production"]

    Publish --> End([End Pipeline])

    Parallel -.->|Log to| MLflow["fa:fa-chart-line MLflow<br/>Track experiments<br/>Store artifacts"]

    style Start fill:#4caf50,color:#fff,stroke:#2e7d32,stroke-width:3px
    style Download fill:#2196f3,color:#fff,stroke:#1565c0,stroke-width:2px
    style Batch fill:#2196f3,color:#fff,stroke:#1565c0,stroke-width:2px
    style Store fill:#ff9800,color:#fff,stroke:#e65100,stroke-width:2px
    style Generate fill:#ffc107,color:#000,stroke:#f57f17,stroke-width:2px
    style Parallel fill:#9c27b0,color:#fff,stroke:#6a1b9a,stroke-width:3px
    style Models fill:#673ab7,color:#fff,stroke:#4527a0,stroke-width:2px
    style Aggregate fill:#3f51b5,color:#fff,stroke:#283593,stroke-width:2px
    style Select fill:#00bcd4,color:#fff,stroke:#006064,stroke-width:2px
    style Publish fill:#4caf50,color:#fff,stroke:#2e7d32,stroke-width:3px
    style End fill:#4caf50,color:#fff,stroke:#2e7d32,stroke-width:3px
    style MLflow fill:#e91e63,color:#fff,stroke:#880e4f,stroke-width:2px
```

### Inference & Serving Architecture

This diagram shows the complete inference infrastructure with API, UI, monitoring, and load testing components.

```mermaid
graph TB
    subgraph "Kubernetes - Minikube Cluster"
        subgraph "Inference Services"
            API1["fa:fa-server API Pod 1<br/>FastAPI<br/>:8000"]
            API2["fa:fa-server API Pod 2<br/>FastAPI<br/>:8000"]
            API3["fa:fa-server API Pod 3<br/>FastAPI<br/>:8000"]
            API_SVC["fa:fa-network-wired API Service<br/>NodePort:30080<br/>LoadBalancer"]
            HPA["fa:fa-chart-line HPA<br/>Auto-scaling<br/>3-10 replicas"]
        end
        
        subgraph "User Interface"
            UI["fa:fa-browser UI Pod<br/>Streamlit<br/>:8501"]
            UI_SVC["fa:fa-network-wired UI Service<br/>NodePort:30085"]
        end
        
        subgraph "Monitoring Stack"
            PROM["fa:fa-chart-bar Prometheus<br/>:9090<br/>Metrics Collection"]
            GRAF["fa:fa-chart-area Grafana<br/>:3000<br/>Dashboards"]
            PROM_SVC["fa:fa-network-wired Prometheus Service<br/>NodePort:30090"]
            GRAF_SVC["fa:fa-network-wired Grafana Service<br/>NodePort:30300"]
        end
        
        subgraph "Load Testing"
            LOCUST["fa:fa-bug Locust<br/>:8089<br/>Load Testing"]
            LOCUST_SVC["fa:fa-network-wired Locust Service<br/>NodePort:30189"]
        end
    end
    
    subgraph "Docker Compose - ML Services"
        MLFLOW["fa:fa-chart-line MLflow<br/>:8002<br/>Model Registry"]
        MINIO["fa:fa-cubes MinIO<br/>:9000<br/>Artifact Store"]
    end
    
    subgraph "External Users"
        USER["fa:fa-user End Users"]
        TESTER["fa:fa-user Load Tester"]
    end
    
    %% User interactions
    USER -->|HTTP Requests| UI_SVC
    USER -->|API Calls| API_SVC
    TESTER -->|Load Test| LOCUST_SVC
    
    %% Service routing
    UI_SVC --> UI
    API_SVC --> API1
    API_SVC --> API2
    API_SVC --> API3
    LOCUST_SVC --> LOCUST
    PROM_SVC --> PROM
    GRAF_SVC --> GRAF
    
    %% API scaling
    HPA --> API1
    HPA --> API2
    HPA --> API3
    
    %% API to MLflow
    API1 -->|Load Model| MLFLOW
    API2 -->|Load Model| MLFLOW
    API3 -->|Load Model| MLFLOW
    MLFLOW --> MINIO
    
    %% Monitoring
    PROM -->|Scrape Metrics| API1
    PROM -->|Scrape Metrics| API2
    PROM -->|Scrape Metrics| API3
    GRAF -->|Query Metrics| PROM
    
    %% Load testing
    LOCUST -->|Generate Load| API_SVC
    
    %% Styling
    style API1 fill:#90caf9,stroke:#1565c0,stroke-width:2px
    style API2 fill:#90caf9,stroke:#1565c0,stroke-width:2px
    style API3 fill:#90caf9,stroke:#1565c0,stroke-width:2px
    style API_SVC fill:#64b5f6,stroke:#0d47a1,stroke-width:3px
    style HPA fill:#81c784,stroke:#2e7d32,stroke-width:2px
    style UI fill:#ffccbc,stroke:#d84315,stroke-width:2px
    style UI_SVC fill:#ffab91,stroke:#bf360c,stroke-width:2px
    style PROM fill:#f48fb1,stroke:#880e4f,stroke-width:2px
    style GRAF fill:#ce93d8,stroke:#6a1b9a,stroke-width:2px
    style LOCUST fill:#fff9c4,stroke:#f57f17,stroke-width:2px
    style MLFLOW fill:#ffccbc,stroke:#e64a19,stroke-width:3px
    style MINIO fill:#ffe0b2,stroke:#f57c00,stroke-width:2px
    style USER fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
    style TESTER fill:#c5e1a5,stroke:#558b2f,stroke-width:2px
```

---

## Step-by-Step Workflow

This project uses a **Cumulative Batch Training Pipeline** to find the optimal training data size by training models on progressively larger datasets.

### Phase 1: Data Acquisition and Preparation

#### 1. Download Dataset

**DAG**: `dag_01_download_data.py`

- Downloads diabetes hospital readmission dataset from Google Drive
- Dataset: ~101,767 patient records with 50+ features
- Stores CSV file in `/opt/airflow/data/diabetic_data.csv`
- Features include: demographics, diagnoses, medications, procedures, lab results

#### 2. Split and Batch Data

**DAG**: `dag_02b_create_batches.py` (part of `dag_00b_master_cumulative_pipeline.py`)

**Split Strategy**:
- **Training**: 70% (~71,237 records)
- **Validation**: 15% (~15,265 records)
- **Test**: 15% (~15,265 records)
- Uses stratified split to maintain class distribution

**Batching Strategy**:
- Training data divided into batches of **15,000 records each**
- Data is shuffled before batching for random distribution
- Expected result: **7 batches** (6 full batches + 1 partial batch)
  - batch_0: 15,000 records
  - batch_1: 15,000 records
  - batch_2: 15,000 records
  - batch_3: 15,000 records
  - batch_4: 15,000 records
  - batch_5: 15,000 records
  - batch_6: ~11,237 records (remaining)

### Phase 2: Data Storage in PostgreSQL

#### 3. Store Batched Raw Data

**DAG**: `dag_03b_store_batched_raw_data.py`

**Actions**:
1. **Create Batch Tables**: Store each batch in separate PostgreSQL tables
   - `batch_0`, `batch_1`, ..., `batch_6` in `raw_data_db`

2. **Create Cumulative Views**: Dynamic SQL views combining progressive batches
   - `train_cumulative_0`: 15,000 records (batch_0)
   - `train_cumulative_1`: 30,000 records (batch_0 + batch_1)
   - `train_cumulative_2`: 45,000 records (batch_0 + batch_1 + batch_2)
   - `train_cumulative_3`: 60,000 records (batches 0-3)
   - `train_cumulative_4`: 75,000 records (batches 0-4)
   - `train_cumulative_5`: 90,000 records (batches 0-5)
   - `train_cumulative_6`: ~101,237 records (all batches)

3. **Store Validation and Test Data**: Store in separate tables
   - `validation_raw` (~15,265 records)
   - `test_raw` (~15,265 records)

**Database Access**:
- **Docker**: `localhost:5433`
- **Kubernetes**: `192.168.58.2:31070`

### Phase 3: Dynamic Training Configuration

#### 4. Generate Training Configurations

**Function**: `generate_training_tasks()` in `dag_00b_master_cumulative_pipeline.py`

**Process**:
1. Pulls cumulative view information from XCom
2. Determines actual number of batches created (N)
3. Generates training configurations for each cumulative dataset
4. Pushes configurations to XCom for parallel training tasks

**Output Example**:
```
Generating Training Configurations
Number of cumulative datasets: 7
  Config 0: 15000 records (train_cumulative_0)
  Config 1: 30000 records (train_cumulative_1)
  Config 2: 45000 records (train_cumulative_2)
  Config 3: 60000 records (train_cumulative_3)
  Config 4: 75000 records (train_cumulative_4)
  Config 5: 90000 records (train_cumulative_5)
  Config 6: 101237 records (train_cumulative_6)
```

### Phase 4: Parallel Cumulative Training

#### 5. Train Models on All Cumulative Datasets

**DAG**: `dag_05b_train_cumulative_batches.py`

**Training Strategy**:
For **each cumulative dataset** (0 through N), train **4 model types**:

1. **Logistic Regression**
   - `max_iter=1000`, `random_state=42`
   - Fast baseline linear classifier

2. **Decision Tree**
   - `max_depth=10`, `random_state=42`
   - Interpretable tree-based model

3. **Random Forest**
   - `n_estimators=100`, `max_depth=10`, `random_state=42`
   - Ensemble bagging method

4. **Gradient Boosting**
   - `n_estimators=100`, `max_depth=5`, `random_state=42`
   - Sequential ensemble boosting

**Total Models Trained**: **7 cumulative datasets × 4 models = 28 models**

**Training Process per Model**:
1. Read data from cumulative view (e.g., `train_cumulative_3`)
2. Handle missing values and clean data
3. Encode categorical features (LabelEncoder)
4. Encode target variable
5. Normalize features (StandardScaler)
6. Train model on cumulative dataset
7. Calculate training metrics
8. Log to MLflow:
   - Parameters: `model_type`, `cumulative_idx`, `n_features`, `n_samples`
   - Metrics: `train_accuracy`, `train_precision`, `train_recall`, `train_f1_score`, `train_roc_auc`
   - Model artifact with preprocessing pipeline

**Parallel Execution**:
- All 7 cumulative training tasks run in **parallel**
- Each task trains 4 models sequentially
- Uses Airflow Celery workers for distributed execution

**MLflow Integration**:
- Experiment: `diabetes_readmission_prediction`
- Tracking Server: `http://localhost:8002`
- Backend Store: PostgreSQL (metadata)
- Artifact Store: MinIO S3 (model files, scalers, encoders)

### Phase 5: Model Selection and Evaluation

#### 6. Aggregate and Select Best Model

**DAG**: `dag_06b_select_best_cumulative_model.py`

**Process**:

**6a. Aggregate Results** (`aggregate_cumulative_results`)
- Collects results from all 28 trained models
- Creates comparison DataFrame with:
  - Model type
  - Cumulative index (data size)
  - Training metrics
  - MLflow run ID
- Sorts by F1 score (descending)
- Analyzes performance patterns across data volumes

**6b. Select Best Model** (`select_best_model`)
- **Primary Criterion**: Highest validation F1 score
- **Secondary Criterion**: ROC-AUC as tiebreaker
- Loads model from MLflow using best run ID
- Evaluates on validation dataset from PostgreSQL
- Logs validation metrics back to MLflow:
  - `val_accuracy`, `val_precision`, `val_recall`, `val_f1_score`, `val_roc_auc`

**6c. Evaluate on Full Validation Set** (`evaluate_on_full_validation`)
- Loads best model from MLflow
- Evaluates on complete `validation_raw` table
- Calculates final performance metrics
- Logs comprehensive evaluation results

**Insights Gained**:
- Optimal training data size (e.g., does 60K perform better than 102K?)
- Best model type for this dataset
- Performance scaling curve with increasing data
- Efficiency vs. accuracy trade-offs

### Phase 6: Production Deployment

#### 7. Publish to Production

**DAG**: `dag_07_publish_to_production.py`

**Deployment Process**:
1. Retrieve best model's MLflow run ID from XCom
2. Register model in **MLflow Model Registry**
   - Model name: `diabetes_readmission_model`
   - Includes metadata and tags
3. **Transition to Production stage**
   - Archives any existing production versions
   - Sets new version as active production model
4. Update model description with:
   - Model type
   - Cumulative dataset size used
   - Validation F1 score
   - Validation accuracy
   - Training timestamp
5. Complete deployment workflow

---

## How to Run the Project

### Prerequisites

1. **Install Docker and Docker Compose**:
   ```bash
   # Verify installation
   docker --version
   docker compose version
   ```

2. **Install Minikube (for Kubernetes deployment)**:
   ```bash
   # Install Minikube
   curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
   sudo install minikube-linux-amd64 /usr/local/bin/minikube

   # Verify installation
   minikube version
   ```

3. **Install kubectl**:
   ```bash
   # Verify installation
   kubectl version --client
   ```

### Step 1: Start Kubernetes Databases (Minikube)

#### 1.1 Start Minikube
```bash
# Start Minikube cluster
minikube start --driver=docker

# Verify cluster is running
minikube status
```

#### 1.2 Connect Docker to Minikube Network
```bash
# Get Minikube Docker network
docker network ls | grep minikube

# If network doesn't exist, create it
docker network create minikube
```

#### 1.3 Deploy PostgreSQL Databases to Kubernetes
```bash
# Navigate to Kubernetes manifests directory
cd k8s/komposedbfiles

# Apply deployments
kubectl apply -f postgres-raw-data-deployment.yaml
kubectl apply -f postgres-clean-data-deployment.yaml

# Apply services
kubectl apply -f postgres-raw-data-service.yaml
kubectl apply -f postgres-clean-data-service.yaml

# Verify deployments
kubectl get deployments
kubectl get pods
kubectl get services
```

#### 1.4 Get Minikube IP and Service Ports
```bash
# Get Minikube IP (typically 192.168.58.2)
minikube ip

# Get NodePort for raw-data database
kubectl get service postgres-raw-data -o jsonpath='{.spec.ports[0].nodePort}'

# Get NodePort for clean-data database
kubectl get service postgres-clean-data -o jsonpath='{.spec.ports[0].nodePort}'

# Alternative: Use minikube service to get URL
minikube service postgres-raw-data --url
minikube service postgres-clean-data --url
```

### Step 2: Start Docker Compose Services

#### 2.1 Set Environment Variables (Optional)
```bash
# Create .env file with custom settings
echo "AIRFLOW_UID=$(id -u)" > .env
```

#### 2.2 Build Custom Docker Images
```bash
# Build Airflow image with ML dependencies
docker compose -f docker-compose.inference.yml build
```

#### 2.3 Start All Services
```bash
# Start all services in detached mode
docker compose -f docker-compose.inference.yml up -d

# View logs
docker compose -f docker-compose.inference.yml logs -f

# Check service health
docker compose -f docker-compose.inference.yml ps
```

#### 2.4 Wait for Initialization
The `airflow-init` service will:
- Create necessary directories
- Initialize Airflow database
- Create default admin user (`airflow:airflow`)

**Wait for**: "airflow-init exited with code 0"

### Step 3: Access Web Interfaces

#### Airflow UI
- **URL**: http://localhost:8001
- **Username**: `airflow`
- **Password**: `airflow`
- **Purpose**: Monitor DAG runs, trigger pipelines, view logs

![Airflow UI](imgs/airflow/airflow.png)
*Apache Airflow UI showing DAGs and pipeline monitoring interface*

#### MLflow UI
- **URL**: http://localhost:8002
- **Purpose**: View experiments, compare models, manage model registry

![MLflow UI](imgs/mlflow-training/mlflow.png)
*MLflow UI showing experiments, model tracking, and model registry*

#### MinIO Console
- **URL**: http://localhost:9001
- **Username**: `admin`
- **Password**: `supersecret`
- **Purpose**: View model artifacts, S3 buckets

### Step 4: Run Cumulative Batch ML Pipeline

#### Run the Master Pipeline (Recommended)

1. **Access Airflow UI**: http://localhost:8001
2. **Login** with credentials:
   - Username: `airflow`
   - Password: `airflow`
3. **Find DAG**: Locate `dag_00b_master_cumulative_pipeline` in the DAG list
4. **Enable DAG**: Toggle the switch to ON (left side of DAG name)
5. **Trigger Pipeline**: Click the play button (▶) on the right to trigger manually
6. **Monitor Progress**:
   - Watch the Graph view to see task dependencies
   - Tasks turn **dark green** when completed successfully
   - Tasks turn **red** if they fail
   - Click on individual tasks to see execution logs
7. **Expected Duration**: ~1-2 hours (trains 28 models in parallel)

#### Pipeline Stages to Monitor

Watch these key stages in the Graph view:

```
Phase 1: Data Acquisition
  ├─ download_diabetes_dataset (2-5 min)
  └─ split_train_val_test (1-2 min)

Phase 2: Batch Storage
  ├─ create_batch_tables (3-5 min)
  ├─ create_cumulative_views (1-2 min)
  └─ store_validation_test (2-3 min)

Phase 3: Dynamic Configuration
  └─ generate_training_configs (1 min)

Phase 4: Parallel Training (15-30 min per cumulative)
  ├─ train_cumulative_0 (4 models on 15K records)
  ├─ train_cumulative_1 (4 models on 30K records)
  ├─ train_cumulative_2 (4 models on 45K records)
  ├─ train_cumulative_3 (4 models on 60K records)
  ├─ train_cumulative_4 (4 models on 75K records)
  ├─ train_cumulative_5 (4 models on 90K records)
  └─ train_cumulative_6 (4 models on ~102K records)

Phase 5: Model Selection
  ├─ aggregate_results (2-3 min)
  ├─ select_best_model (3-5 min)
  └─ evaluate_on_full_validation (2-3 min)

Phase 6: Production Deployment
  └─ publish_best_model_production (2-3 min)
```

#### Alternative: Run Individual DAGs (Advanced)

For debugging or testing individual components, execute DAGs sequentially:

```bash
# Phase 1: Data Acquisition
dag_01_download_data
    ↓
# Phase 2: Batch Creation and Storage
dag_02b_create_batches
    ↓
dag_03b_store_batched_raw_data
    ↓
# Phase 3 & 4: Training (Run after config generation)
dag_05b_train_cumulative_batches
    ↓
# Phase 5: Model Selection
dag_06b_select_best_cumulative_model
    ↓
# Phase 6: Deployment
dag_07_publish_to_production
```

**Note**: When running individual DAGs, ensure each completes successfully before running the next.

### Step 5: View Results

#### MLflow Experiments
1. Open MLflow UI: http://localhost:8002
2. Navigate to "Experiments" → `diabetes_readmission_prediction`
3. Compare runs by metrics (F1 score, accuracy, etc.)
4. View model parameters and artifacts

![MLflow Experiments](imgs/mlflow-training/mlflow.png)
*MLflow UI showing experiment tracking, model comparisons, and performance metrics*

#### Training Metrics Visualization

The following visualizations show the training and validation metrics across all 28 models (7 cumulative datasets × 4 algorithms):

**Training Metrics:**

**Training Accuracy:**
![Training Accuracy](imgs/mlflow-training/train_accuracy.png)

**Training F1 Score:**
![Training F1 Score](imgs/mlflow-training/train_f1_score.png)

**Training Precision:**
![Training Precision](imgs/mlflow-training/train_precision.png)

**Training Recall:**
![Training Recall](imgs/mlflow-training/train_recall.png)

**Validation Metrics:**

**Validation Accuracy:**
![Validation Accuracy](imgs/mlflow-training/val_accuracy.png)

**Validation F1 Score:**
![Validation F1 Score](imgs/mlflow-training/val_f1_score.png)

**Validation Precision:**
![Validation Precision](imgs/mlflow-training/val_precision.png)

**Validation Recall:**
![Validation Recall](imgs/mlflow-training/val_recall.png)

These visualizations help identify:
- Optimal training data size (which cumulative dataset performs best)
- Best performing algorithm for this dataset
- Performance scaling with increasing data volume
- Trade-offs between different metrics

#### MLflow Model Registry
1. Navigate to "Models" in MLflow UI
2. Find `diabetes_readmission_model`
3. View production version and metadata
4. Download model artifacts

### Step 5.5: Jupyter Notebooks (Optional)

For data exploration and analysis, Jupyter notebooks are available:

#### 5.5.1 Start Jupyter Service
```bash
# Navigate to jupyter directory
cd jupyter

# Start Jupyter with Docker Compose
docker compose -f docker-compose.jupyter.yml up -d

# Access Jupyter Lab
# URL: http://localhost:8888
# Token: Check logs with: docker compose -f docker-compose.jupyter.yml logs
```

#### 5.5.2 Available Notebooks
- Data exploration and analysis
- Model evaluation and comparison
- Feature importance analysis
- Performance visualization

#### 5.5.3 Stop Jupyter Service
```bash
cd jupyter
docker compose -f docker-compose.jupyter.yml down
```

#### Database Verification
```bash
# Connect to raw data database
docker exec -it raw_data psql -U rawdata_user -d raw_data_db

# List tables
\dt

# Query training data
SELECT COUNT(*) FROM train_raw;

# Exit
\q

# Connect to clean data database
docker exec -it clean_data psql -U cleandata_user -d clean_data_db

# Query cleaned data
SELECT COUNT(*) FROM train_clean;
\q
```

### Step 6: Stop Services

#### Stop Docker Compose
```bash
# Stop all services
docker compose -f docker-compose.inference.yml down

# Stop and remove volumes (WARNING: deletes all data)
docker compose -f docker-compose.inference.yml down -v
```

#### Stop Kubernetes Services
```bash
# Delete Kubernetes resources
kubectl delete -f k8s/komposedbfiles/

# Stop Minikube
minikube stop

# Delete Minikube cluster (optional)
minikube delete
```

### Step 7: Deploy Inference Services (API, UI, Monitoring)

After training and registering a model in MLflow, deploy the inference infrastructure to Kubernetes.

#### 7.1 Enable Metrics Server (Required for HPA)
```bash
# Enable metrics server in Minikube
minikube addons enable metrics-server

# Verify metrics server is running
kubectl get deployment metrics-server -n kube-system
```

#### 7.2 Deploy All Services (Recommended)
```bash
# Navigate to scripts directory
cd k8s/scripts

# Deploy API, UI, Monitoring, and Locust
./deploy.sh

# Or deploy individual components:
./deploy-api.sh      # Deploy API only
./deploy-ui.sh       # Deploy UI only
./deploy-monitoring.sh  # Deploy Prometheus & Grafana
./deploy-locust.sh   # Deploy Locust load testing
```

#### 7.3 Verify Deployments
```bash
# Check all deployments
kubectl get deployments

# Check all services
kubectl get services

# Check API pods
kubectl get pods -l app=diabetes-api

# Check HPA status
kubectl get hpa diabetes-api-hpa

# View API logs
kubectl logs -l app=diabetes-api -f
```

#### 7.4 Access Services

**Get Minikube IP**:
```bash
minikube ip
# Expected output: 192.168.49.2 (or similar)
```

**API Service**:
- **NodePort**: `http://<minikube-ip>:30080`
- **Health Check**: `http://<minikube-ip>:30080/health`
- **API Docs**: `http://<minikube-ip>:30080/docs`
- **Model Info**: `http://<minikube-ip>:30080/model-info`
- **Metrics**: `http://<minikube-ip>:30080/metrics`

**UI Service**:
- **URL**: `http://<minikube-ip>:30085`

**Prometheus**:
- **URL**: `http://<minikube-ip>:30090`

**Grafana**:
- **URL**: `http://<minikube-ip>:30300`
- **Default Credentials**: `admin:admin` (change on first login)

**Locust**:
- **URL**: `http://<minikube-ip>:30189`

#### 7.5 Port Forwarding (Alternative Access)

For easier access from your local machine, use port forwarding:

```bash
# Start port forwarding script
cd k8s/scripts
./start-port-forward.sh

# This forwards:
# - Port 8000 → API (30080)
# - Port 8010 → UI (30085)
# - Port 8011 → Prometheus (30090)
# - Port 8012 → Grafana (30300)
# - Port 8013 → Locust (30189)

# Access services on localhost:
# - API: http://localhost:8000
# - UI: http://localhost:8010
# - Prometheus: http://localhost:8011
# - Grafana: http://localhost:8012
# - Locust: http://localhost:8013
```

### Step 8: Using the Inference API

#### 8.1 Health Check
```bash
curl http://<minikube-ip>:30080/health
# Expected response: {"status":"ok","model":"diabetes_readmission_model"}
```

#### 8.2 Get Model Information
```bash
curl http://<minikube-ip>:30080/model-info
```

#### 8.3 Make Predictions

**Single Prediction**:
```bash
curl -X POST "http://<minikube-ip>:30080/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "race": "Caucasian",
      "gender": "Female",
      "age": "[70-80)",
      "weight": "?",
      "admission_type_id": 1,
      "discharge_disposition_id": 1,
      "admission_source_id": 7,
      "time_in_hospital": 1,
      "payer_code": "?",
      "medical_specialty": "Emergency/Trauma",
      "num_lab_procedures": 41,
      "num_procedures": 0,
      "num_medications": 1,
      "number_outpatient": 0,
      "number_emergency": 0,
      "number_inpatient": 0,
      "diag_1": "250.83",
      "diag_2": "?",
      "diag_3": "?",
      "number_diagnoses": 1,
      "max_glu_serum": "None",
      "A1Cresult": "None",
      "metformin": "No",
      "repaglinide": "No",
      "nateglinide": "No",
      "chlorpropamide": "No",
      "glimepiride": "No",
      "acetohexamide": "No",
      "glipizide": "No",
      "glyburide": "No",
      "tolbutamide": "No",
      "pioglitazone": "No",
      "rosiglitazone": "No",
      "acarbose": "No",
      "miglitol": "No",
      "troglitazone": "No",
      "tolazamide": "No",
      "examide": "No",
      "citoglipton": "No",
      "insulin": "No",
      "glyburide-metformin": "No",
      "glipizide-metformin": "No",
      "glimepiride-pioglitazone": "No",
      "metformin-rosiglitazone": "No",
      "metformin-pioglitazone": "No",
      "change": "No",
      "diabetesMed": "No"
    }
  }'
```

**Expected Response**:
```json
{
  "readmission_prediction": "NO",
  "probabilities": {
    "NO": 0.85,
    "<30": 0.10,
    ">30": 0.05
  }
}
```

#### 8.4 Python Client Example
```python
import requests
import json

API_URL = "http://<minikube-ip>:30080"

# Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# Get model info
response = requests.get(f"{API_URL}/model-info")
print(json.dumps(response.json(), indent=2))

# Make prediction
patient_data = {
    "features": {
        "race": "Caucasian",
        "gender": "Female",
        "age": "[70-80)",
        "admission_type_id": 1,
        "discharge_disposition_id": 1,
        "time_in_hospital": 1,
        "num_lab_procedures": 41,
        "num_medications": 1,
        # ... (add all required features)
    }
}

response = requests.post(f"{API_URL}/predict", json=patient_data)
result = response.json()
print(f"Prediction: {result['readmission_prediction']}")
print(f"Probabilities: {result['probabilities']}")
```

#### 8.5 API Documentation (Swagger UI)

Access the interactive API documentation at `http://<minikube-ip>:30080/docs`:

![API Swagger Documentation](imgs/api/api_swagger.png)
*FastAPI Swagger UI showing interactive API documentation and endpoint testing interface*

#### 8.6 API Endpoints Summary

**Base URL**: `http://<minikube-ip>:30080`

| Endpoint | Method | Description | Response |
|----------|--------|-------------|----------|
| `/health` | GET | Health check endpoint | `{"status": "ok", "model": "diabetes_readmission_model"}` |
| `/model-info` | GET | Get model information | Model metadata, version, features, preprocessing status |
| `/predict` | POST | Make prediction | `{"readmission_prediction": "NO", "probabilities": {...}}` |
| `/metrics` | GET | Prometheus metrics | Prometheus-formatted metrics |
| `/docs` | GET | API documentation | Swagger UI interactive docs |
| `/openapi.json` | GET | OpenAPI specification | JSON schema |

**Request Format**:
```json
{
  "features": {
    "race": "Caucasian",
    "gender": "Female",
    "age": "[70-80)",
    // ... (50+ features)
  }
}
```

**Response Format**:
```json
{
  "readmission_prediction": "NO",
  "probabilities": {
    "NO": 0.85,
    "<30": 0.10,
    ">30": 0.05
  }
}
```

**Prediction Classes**:
- `NO`: No readmission
- `<30`: Readmission within 30 days
- `>30`: Readmission after 30 days

### Step 9: Using the Web UI

#### 9.1 Access the UI
1. Open your browser and navigate to: `http://<minikube-ip>:30085`
2. The UI will display:
   - Model information (version, type, metrics)
   - Interactive form for patient data entry
   - Prediction results with probability distributions
   - Model performance metrics

![Web UI](imgs/ui/ui.png)
*Streamlit Web UI showing model information, patient data entry form, and prediction results*

#### 9.2 Making Predictions via UI
1. Fill in the patient information form
2. Click "Predict Readmission"
3. View the prediction result and probability distribution
4. Optionally, view model metadata and performance metrics

### Step 10: Monitoring with Prometheus & Grafana

#### 10.1 Access Prometheus
1. Open Prometheus UI: `http://<minikube-ip>:30090`
2. Navigate to "Targets" to verify API metrics are being scraped
3. Use "Graph" to query metrics:
   - `predict_requests_total` - Total prediction requests
   - `predict_latency_seconds` - Request latency
   - `predictions_total` - Predictions by class
   - `predict_errors_total` - Error count by type
   - `predict_requests_in_progress` - Current in-progress requests

#### 10.2 Access Grafana
1. Open Grafana UI: `http://<minikube-ip>:30300`
2. Login with default credentials: `admin:admin`
3. Add Prometheus data source:
   - URL: `http://prometheus:9090` (internal) or `http://<minikube-ip>:30090` (external)
   - Access: Server (default)
4. Create dashboards for:
   - API request rate and latency
   - Prediction distribution
   - Error rates
   - Pod resource usage
   - HPA scaling events

#### 10.3 Key Metrics to Monitor
- **Request Rate**: Requests per second
- **Latency**: P50, P95, P99 response times
- **Error Rate**: Percentage of failed requests
- **Prediction Distribution**: Distribution of prediction classes
- **Resource Usage**: CPU and memory usage per pod
- **Scaling Events**: HPA scaling up/down events

#### 10.4 Grafana Dashboard Screenshots

**Overview Dashboard - During Active Load Testing:**
![Overview Dashboard - Active Load](imgs/observability/overview_last5minutes_locustActive.png)
*Dashboard showing API performance metrics during active load testing with Locust*

**Overview Dashboard - After Load Test Stopped:**
![Overview Dashboard - Load Stopped](imgs/observability/overview_last15minutes_locustStopped.png)
*Dashboard showing API performance metrics after load test stopped*

**Predictions Dashboard - During Active Load Testing:**
![Predictions Dashboard - Active Load](imgs/observability/predictions_last5minutes_locustActive.png)
*Dashboard showing prediction distribution and class metrics during active load testing*

**Predictions Dashboard - After Load Test Stopped:**
![Predictions Dashboard - Load Stopped](imgs/observability/predictions_last15minutes_locustStopped.png)
*Dashboard showing prediction distribution and class metrics after load test stopped*

These dashboards provide real-time insights into:
- API performance under load
- Request rates and latency patterns
- Prediction class distribution
- System resource utilization
- Error rates and failure patterns

### Step 11: Load Testing with Locust

#### 11.1 Access Locust UI
1. Open Locust UI: `http://<minikube-ip>:30189`
2. Configure test parameters:
   - **Number of users**: 10-100 (concurrent users)
   - **Spawn rate**: 2 (users per second)
   - **Host**: `http://diabetes-api:8000` (internal) or `http://<minikube-ip>:30080` (external)
3. Click "Start swarming" to begin load test

#### 11.2 Monitor Load Test
- View real-time statistics:
  - Total requests per second
  - Response times (min, max, average, median)
  - Number of failures
  - Response time distribution
- Observe HPA scaling:
  ```bash
  # Watch HPA scale up/down
  watch kubectl get hpa diabetes-api-hpa
  
  # Watch pod count
  watch kubectl get pods -l app=diabetes-api
  ```

#### 11.3 Load Test Scenarios
- **Light Load**: 10 users, 2 spawn rate
- **Medium Load**: 50 users, 5 spawn rate
- **Heavy Load**: 100 users, 10 spawn rate
- **Stress Test**: 200+ users, 20 spawn rate

#### 11.4 Load Testing Results

**Stable Performance at 8,500 Users:**
![8500 Users Stable](imgs/locust/8500users_stable.png)
*Locust load test results showing stable performance at 8,500 concurrent users*

The system demonstrates stable performance with:
- Consistent response times
- Low error rates
- Successful request handling
- Proper HPA scaling behavior

**System Crash at 9,000 Users:**
![9000 Users Crashed](imgs/locust/9000users_crashed.png)
*Locust load test results showing system crash at 9,000 concurrent users*

At 9,000 concurrent users, the system reaches its breaking point:
- Increased error rates
- Response time degradation
- System instability
- Resource exhaustion

**Key Findings:**
- **Maximum Stable Load**: ~8,500 concurrent users
- **Breaking Point**: ~9,000 concurrent users
- **Optimal Configuration**: 3-10 replicas with HPA enabled
- **Resource Limits**: CPU and memory thresholds are critical for stability

#### 11.5 Analyze Results
- Check Prometheus metrics during load test
- Verify HPA scales appropriately
- Monitor API pod resource usage
- Check for errors in API logs
- Compare performance at different load levels
- Identify system bottlenecks and capacity limits

### Step 12: Undeploy Services

#### 12.1 Undeploy All Services
```bash
cd k8s/scripts
./undeploy.sh --all
```

#### 12.2 Undeploy Individual Services
```bash
./undeploy.sh --api           # Undeploy API only
./undeploy.sh --ui            # Undeploy UI only
./undeploy.sh --monitoring    # Undeploy Monitoring only
./undeploy.sh --locust        # Undeploy Locust only
```

---