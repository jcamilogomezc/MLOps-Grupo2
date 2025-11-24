# Step-by-Step Guide to Start Server with Helm and Minikube

## Start Server

```bash
minikube start --cpus=4 --memory=4096 --addons=metallb --extra-config=apiserver.service-node-port-range=1-65535
```

## Create Namespace

```bash
kubectl create namespace airflow
```

## Adding the Apache Airflow Helm Chart Repository

```bash
helm repo add apache-airflow https://airflow.apache.org
```

## Installing Apache Airflow Using Helm

```bash
helm install airflow apache-airflow/airflow --namespace airflow --debug --timeout 10m01s -f airflow/charts/values.yaml
```

## Configuration to Expose Airflow API Server (Optional)

### Update Type of Airflow Server to Use LoadBalancer

```bash
kubectl apply -f ./airflow/api-server-lb.yaml
```

### Configure MetalLB

```bash
kubectl apply -f ./airflow/metallb-config.yaml
```

### Expose with Port Forward

```bash
kubectl port-forward service/airflow-api-server 8080:8080 --namespace airflow
```

## Get Cluster Info

```bash
kubectl get all --all-namespaces
```

## Cleanup

```bash
minikube delete --all --purge
```

---

## Argo Workflows

### Install Argo

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

## Build and Deploy to DockerHub

```bash
AIRFLOW_USER="germanaoq"

docker build -t "${AIRFLOW_USER}/airflow-custom:latest" .
docker push "${AIRFLOW_USER}/airflow-custom:latest"
```

## References

- [Helm Basics: Understanding Charts, Templates, and Repositories](https://medium.com/@bavicnative/helm-basics-understanding-charts-templates-and-repositories-6b8e55f539e0)
- [Deploying Apache Airflow on Kubernetes with ArgoCD: A GitOps Approach](https://medium.com/@farman.bsse1855/deploying-apache-airflow-on-kubernetes-with-argocd-a-gitops-approach-9e4ea89f10fa)
- [Minikube K8s Service External IP Pending](https://www.baeldung.com/ops/minikube-k8s-service-external-ip-pending)
