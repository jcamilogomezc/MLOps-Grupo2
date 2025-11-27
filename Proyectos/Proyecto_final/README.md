# Step-by-Step Guide to Deploy MLOps Infrastructure with Helm and Minikube

## Start Minikube Server

```bash
minikube start --cpus=5 --memory=7851 --addons=metallb --extra-config=apiserver.service-node-port-range=1-65535
```

## Create MLOps Namespace

```bash
kubectl create namespace mlops
```

## Apache Airflow

### Add the Apache Airflow Helm Chart Repository

```bash
helm repo add apache-airflow https://airflow.apache.org
```

### Install Apache Airflow Using Helm

```bash
helm install airflow apache-airflow/airflow --namespace mlops --debug --timeout 10m01s -f airflow/charts/values.yaml
```

### Configuration to Expose Airflow API Server (Optional)

#### Update Airflow Server Type to Use LoadBalancer

```bash
kubectl apply -f ./airflow/api-server-lb.yaml
```

#### Configure MetalLB

```bash
kubectl apply -f ./airflow/metallb-config.yaml
```

#### Expose with Port Forward

```bash
kubectl port-forward service/airflow-api-server 8080:8080 --namespace mlops
```

### Expose with tunnel minikube
```bash
minikube service airflow-api-server -n mlops
```

## Get Cluster Info

```bash
kubectl get all --all-namespaces
```

---

## MLflow

### Add Bitnami Repository and Install MLflow

```bash
helm repo add bitnami https://charts.bitnami.com/bitnami
helm repo update

helm install mlflow community-charts/mlflow -f ./mlflow/values.yaml --namespace mlops --debug --timeout 10m01s
```

### Expose with tunnel minikube
```bash
minikube service mlflow -n mlops
```

---


## PostgreSQL Raw

### Install PostgreSQL for Raw Data

```bash
helm install psql-raw bitnami/postgresql --namespace mlops -f postgresql/values-raw.yml
```

---

## PostgreSQL Clean

### Install PostgreSQL for Clean Data

```bash
helm install psql-clean bitnami/postgresql --namespace mlops -f postgresql/values-clean.yml
```

---

## Inference API

### Install
```bash
helm install inference-api ./api --namespace mlops --debug --timeout 10m01s
```

---

## Argo Workflows

### Install Argo Workflows

Specify the version:

```bash
ARGO_WORKFLOWS_VERSION="v3.7.4"
```

Apply the quick-start manifest:

```bash
kubectl create namespace argo
kubectl apply -n argo -f "https://github.com/argoproj/argo-workflows/releases/download/${ARGO_WORKFLOWS_VERSION}/quick-start-minimal.yaml"
```

---

## Cleanup

```bash
minikube delete --all --purge
```

## References

- [Helm Basics: Understanding Charts, Templates, and Repositories](https://medium.com/@bavicnative/helm-basics-understanding-charts-templates-and-repositories-6b8e55f539e0)
- [Deploying Apache Airflow on Kubernetes with ArgoCD: A GitOps Approach](https://medium.com/@farman.bsse1855/deploying-apache-airflow-on-kubernetes-with-argocd-a-gitops-approach-9e4ea89f10fa)
- [Minikube K8s Service External IP Pending](https://www.baeldung.com/ops/minikube-k8s-service-external-ip-pending)
