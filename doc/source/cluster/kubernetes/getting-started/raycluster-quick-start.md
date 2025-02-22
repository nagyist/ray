(kuberay-raycluster-quickstart)=

# RayCluster Quickstart

This guide shows you how to manage and interact with Ray clusters on Kubernetes.

## Preparation

* Install [kubectl](https://kubernetes.io/docs/tasks/tools/#kubectl) (>= 1.23), [Helm](https://helm.sh/docs/intro/install/) (>= v3.4) if needed, [Kind](https://kind.sigs.k8s.io/docs/user/quick-start/#installation), and [Docker](https://docs.docker.com/engine/install/).
* Make sure your Kubernetes cluster has at least 4 CPU and 4 GB RAM.

## Step 1: Create a Kubernetes cluster

This step creates a local Kubernetes cluster using [Kind](https://kind.sigs.k8s.io/). If you already have a Kubernetes cluster, you can skip this step.

```sh
kind create cluster --image=kindest/node:v1.26.0
```

(kuberay-operator-deploy)=
## Step 2: Deploy a KubeRay operator

Deploy the KubeRay operator with the [Helm chart repository](https://github.com/ray-project/kuberay-helm) or Kustomize.

`````{tab-set}

````{tab-item} Helm

```sh
helm repo add kuberay https://ray-project.github.io/kuberay-helm/
helm repo update

# Install both CRDs and KubeRay operator v1.3.0.
helm install kuberay-operator kuberay/kuberay-operator --version 1.3.0

# Confirm that the operator is running in the namespace `default`.
kubectl get pods
# NAME                                READY   STATUS    RESTARTS   AGE
# kuberay-operator-7fbdbf8c89-pt8bk   1/1     Running   0          27s
```

````

````{tab-item} Kustomize

```sh
# Install CRD and KubeRay operator.
kubectl create -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.3.0"

# Confirm that the operator is running.
kubectl get pods
# NAME                                READY   STATUS    RESTARTS   AGE
# kuberay-operator-6d57c9f797-ffvph   1/1     Running   0          2m14s

```

````

`````

For further information, see [the installation instructions in the KubeRay documentation](https://ray-project.github.io/kuberay/deploy/installation/).

(raycluster-deploy)=
## Step 3: Deploy a RayCluster custom resource

Once the KubeRay operator is running, you're ready to deploy a RayCluster. Create a RayCluster Custom Resource (CR) in the `default` namespace.

  ::::{tab-set}

  :::{tab-item} Helm ARM64 (Apple Silicon)
  ```sh
  # Deploy a sample RayCluster CR from the KubeRay Helm chart repo:
  helm install raycluster kuberay/ray-cluster --version 1.3.0 --set 'image.tag=2.41.0-aarch64'
  ```
  :::

  :::{tab-item} Helm x86-64 (Intel/Linux)
  ```sh
  # Deploy a sample RayCluster CR from the KubeRay Helm chart repo:
  helm install raycluster kuberay/ray-cluster --version 1.3.0
  ```
  :::

  :::{tab-item} Kustomize
  ```sh
  # Deploy a sample RayCluster CR from the KubeRay repository:
  kubectl apply -f "https://raw.githubusercontent.com/ray-project/kuberay/v1.3.0/ray-operator/config/samples/ray-cluster.sample.yaml"
  ```
  :::

  ::::


```sh
# Once the RayCluster CR has been created, you can view it by running:
kubectl get rayclusters

# NAME                 DESIRED WORKERS   AVAILABLE WORKERS   CPUS   MEMORY   GPUS   STATUS   AGE
# raycluster-kuberay   1                 1                   2      3G       0      ready    95s
```

The KubeRay operator detects the RayCluster object and starts your Ray cluster by creating head and worker pods. To view Ray cluster's pods, run the following command:

```sh
# View the pods in the RayCluster named "raycluster-kuberay"
kubectl get pods --selector=ray.io/cluster=raycluster-kuberay

# NAME                                          READY   STATUS    RESTARTS   AGE
# raycluster-kuberay-head-vkj4n                 1/1     Running   0          XXs
# raycluster-kuberay-worker-workergroup-xvfkr   1/1     Running   0          XXs
```

Wait for the pods to reach Running state. This may take a few minutes, downloading the Ray images takes most of this time.
If your pods stick in the Pending state, you can check for errors using `kubectl describe pod raycluster-kuberay-xxxx-xxxxx` and ensure your Docker resource limits meet the requirements.

## Step 4: Run an application on a RayCluster

Now, interact with the RayCluster deployed.

### Method 1: Execute a Ray job in the head Pod

The most straightforward way to experiment with your RayCluster is to exec directly into the head pod.
First, identify your RayCluster's head pod:

```sh
export HEAD_POD=$(kubectl get pods --selector=ray.io/node-type=head -o custom-columns=POD:metadata.name --no-headers)
echo $HEAD_POD
# raycluster-kuberay-head-vkj4n

# Print the cluster resources.
kubectl exec -it $HEAD_POD -- python -c "import ray; ray.init(); print(ray.cluster_resources())"

# 2023-04-07 10:57:46,472 INFO worker.py:1243 -- Using address 127.0.0.1:6379 set in the environment variable RAY_ADDRESS
# 2023-04-07 10:57:46,472 INFO worker.py:1364 -- Connecting to existing Ray cluster at address: 10.244.0.6:6379...
# 2023-04-07 10:57:46,482 INFO worker.py:1550 -- Connected to Ray cluster. View the dashboard at http://10.244.0.6:8265
# {'object_store_memory': 802572287.0, 'memory': 3000000000.0, 'node:10.244.0.6': 1.0, 'CPU': 2.0, 'node:10.244.0.7': 1.0}
```

### Method 2: Submit a Ray job to the RayCluster using [ray job submission SDK](jobs-quickstart)

Unlike Method 1, this method doesn't require you to execute commands in the Ray head pod.
Instead, you can use the [Ray job submission SDK](jobs-quickstart) to submit Ray jobs to the RayCluster through the Ray Dashboard port where Ray listens for Job requests.
The KubeRay operator configures a [Kubernetes service](https://kubernetes.io/docs/concepts/services-networking/service/) targeting the Ray head Pod.

```sh
kubectl get service raycluster-kuberay-head-svc

# NAME                          TYPE        CLUSTER-IP    EXTERNAL-IP   PORT(S)                                         AGE
# raycluster-kuberay-head-svc   ClusterIP   10.96.93.74   <none>        8265/TCP,8080/TCP,8000/TCP,10001/TCP,6379/TCP   15m
```

Now that the service name is available, use port-forwarding to access the Ray Dashboard port which is 8265 by default.

```sh
# Execute this in a separate shell.
kubectl port-forward service/raycluster-kuberay-head-svc 8265:8265
```

Now that the Dashboard port is accessible, submit jobs to the RayCluster:

```sh
# The following job's logs will show the Ray cluster's total resource capacity, including 2 CPUs.
ray job submit --address http://localhost:8265 -- python -c "import ray; ray.init(); print(ray.cluster_resources())"
```

## Step 5: Access the Ray Dashboard

Visit `${YOUR_IP}:8265` in your browser for the Dashboard. For example, `127.0.0.1:8265`.
See the job you submitted in Step 4 in the **Recent jobs** pane as shown below.

![Ray Dashboard](../images/ray-dashboard.png)

## Step 6: Cleanup

`````{tab-set}

````{tab-item} Helm

```sh
# [Step 6.1]: Delete the RayCluster CR
# Uninstall the RayCluster Helm chart
helm uninstall raycluster
# release "raycluster" uninstalled

# Note that it may take several seconds for the Ray pods to be fully terminated.
# Confirm that the RayCluster's pods are gone by running
kubectl get pods

# NAME                                READY   STATUS    RESTARTS   AGE
# kuberay-operator-7fbdbf8c89-pt8bk   1/1     Running   0          XXm

# [Step 6.2]: Delete the KubeRay operator
# Uninstall the KubeRay operator Helm chart
helm uninstall kuberay-operator
# release "kuberay-operator" uninstalled

# Confirm that the KubeRay operator pod is gone by running
kubectl get pods
# No resources found in default namespace.

# [Step 6.3]: Delete the Kubernetes cluster
kind delete cluster
```

````

````{tab-item} Kustomize

```sh
# [Step 6.1]: Delete the RayCluster CR
kubectl delete -f "https://raw.githubusercontent.com/ray-project/kuberay/v1.3.0/ray-operator/config/samples/ray-cluster.sample.yaml"

# Confirm that the RayCluster's pods are gone by running
kubectl get pods
# NAME                                READY   STATUS    RESTARTS   AGE
# kuberay-operator-7fbdbf8c89-pt8bk   1/1     Running   0          XXm

# [Step 6.2]: Delete the KubeRay operator
kubectl delete -k "github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.3.0"

# Confirm that the KubeRay operator pod is gone by running
kubectl get pods
# No resources found in the default namespace.

# [Step 6.3]: Delete the Kubernetes cluster
kind delete cluster
```

````

`````
