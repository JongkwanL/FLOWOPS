"""Deployment pipeline for model serving and orchestration."""

import os
import json
import yaml
import argparse
import requests
import time
from typing import Dict, Any, Optional, List
from datetime import datetime
import kubernetes
from kubernetes import client, config
import mlflow
from mlflow.tracking import MlflowClient
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelDeployer:
    """Handles model deployment to Kubernetes with progressive delivery."""
    
    def __init__(self, kubeconfig_path: Optional[str] = None):
        """Initialize deployer with Kubernetes configuration."""
        try:
            if kubeconfig_path:
                config.load_kube_config(config_file=kubeconfig_path)
            else:
                config.load_incluster_config()  # For in-cluster deployment
        except:
            config.load_kube_config()  # Default kubeconfig
        
        self.k8s_apps = client.AppsV1Api()
        self.k8s_core = client.CoreV1Api()
        self.k8s_custom = client.CustomObjectsApi()
        self.mlflow_client = MlflowClient()
    
    def get_latest_model_version(self, model_name: str, stage: str = "Production") -> Dict[str, Any]:
        """Get latest model version from MLflow registry."""
        try:
            # Search for model versions
            filter_string = f"name='{model_name}'"
            versions = self.mlflow_client.search_model_versions(filter_string)
            
            # Filter by stage
            staged_versions = [v for v in versions if v.current_stage == stage]
            
            if not staged_versions:
                logger.warning(f"No model found in {stage} stage")
                return None
            
            # Get latest version
            latest = max(staged_versions, key=lambda x: int(x.version))
            
            return {
                'name': latest.name,
                'version': latest.version,
                'stage': latest.current_stage,
                'run_id': latest.run_id,
                'source': latest.source,
                'tags': latest.tags
            }
        except Exception as e:
            logger.error(f"Error fetching model from registry: {e}")
            return None
    
    def create_deployment_manifest(
        self,
        model_info: Dict[str, Any],
        namespace: str = "production",
        replicas: int = 3,
        strategy: str = "canary"
    ) -> Dict[str, Any]:
        """Create Kubernetes deployment manifest for model serving."""
        
        # Container image tag based on model version
        image_tag = f"{model_info['run_id'][:8]}-v{model_info['version']}"
        
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": f"model-{model_info['name']}-v{model_info['version']}",
                "namespace": namespace,
                "labels": {
                    "app": "flowops",
                    "component": "model-serving",
                    "model": model_info['name'],
                    "version": str(model_info['version']),
                    "stage": model_info['stage']
                }
            },
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {
                        "app": "flowops",
                        "model": model_info['name']
                    }
                },
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "flowops",
                            "model": model_info['name'],
                            "version": str(model_info['version'])
                        },
                        "annotations": {
                            "prometheus.io/scrape": "true",
                            "prometheus.io/port": "8000",
                            "prometheus.io/path": "/metrics"
                        }
                    },
                    "spec": {
                        "securityContext": {
                            "runAsNonRoot": True,
                            "runAsUser": 1000,
                            "fsGroup": 1000
                        },
                        "containers": [{
                            "name": "model-serving",
                            "image": f"ghcr.io/flowops/mlops-model:{image_tag}-serving",
                            "imagePullPolicy": "IfNotPresent",
                            "ports": [
                                {"containerPort": 8080, "name": "http"},
                                {"containerPort": 8000, "name": "metrics"}
                            ],
                            "env": [
                                {"name": "MODEL_NAME", "value": model_info['name']},
                                {"name": "MODEL_VERSION", "value": str(model_info['version'])},
                                {"name": "MODEL_STAGE", "value": model_info['stage']},
                                {"name": "MLFLOW_TRACKING_URI", "valueFrom": {
                                    "secretKeyRef": {
                                        "name": "mlflow-secrets",
                                        "key": "tracking_uri"
                                    }
                                }},
                                {"name": "LOG_LEVEL", "value": "INFO"}
                            ],
                            "resources": {
                                "requests": {
                                    "memory": "1Gi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "4Gi",
                                    "cpu": "2000m",
                                    "nvidia.com/gpu": "0"  # Set to 1 for GPU
                                }
                            },
                            "livenessProbe": {
                                "httpGet": {
                                    "path": "/health",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 30,
                                "periodSeconds": 10,
                                "timeoutSeconds": 5,
                                "failureThreshold": 3
                            },
                            "readinessProbe": {
                                "httpGet": {
                                    "path": "/ready",
                                    "port": 8080
                                },
                                "initialDelaySeconds": 10,
                                "periodSeconds": 5,
                                "timeoutSeconds": 3,
                                "failureThreshold": 3
                            },
                            "securityContext": {
                                "allowPrivilegeEscalation": False,
                                "readOnlyRootFilesystem": True,
                                "capabilities": {
                                    "drop": ["ALL"]
                                }
                            },
                            "volumeMounts": [
                                {
                                    "name": "tmp",
                                    "mountPath": "/tmp"
                                },
                                {
                                    "name": "cache",
                                    "mountPath": "/home/mluser/.cache"
                                }
                            ]
                        }],
                        "volumes": [
                            {
                                "name": "tmp",
                                "emptyDir": {}
                            },
                            {
                                "name": "cache",
                                "emptyDir": {}
                            }
                        ]
                    }
                }
            }
        }
        
        # Add strategy-specific configurations
        if strategy == "rolling":
            manifest["spec"]["strategy"] = {
                "type": "RollingUpdate",
                "rollingUpdate": {
                    "maxSurge": 1,
                    "maxUnavailable": 0
                }
            }
        
        return manifest
    
    def create_service_manifest(
        self,
        model_info: Dict[str, Any],
        namespace: str = "production"
    ) -> Dict[str, Any]:
        """Create Kubernetes service manifest."""
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {
                "name": f"model-{model_info['name']}-service",
                "namespace": namespace,
                "labels": {
                    "app": "flowops",
                    "model": model_info['name']
                }
            },
            "spec": {
                "selector": {
                    "app": "flowops",
                    "model": model_info['name']
                },
                "ports": [
                    {
                        "name": "http",
                        "port": 80,
                        "targetPort": 8080,
                        "protocol": "TCP"
                    },
                    {
                        "name": "metrics",
                        "port": 8000,
                        "targetPort": 8000,
                        "protocol": "TCP"
                    }
                ],
                "type": "ClusterIP"
            }
        }
    
    def create_hpa_manifest(
        self,
        model_info: Dict[str, Any],
        namespace: str = "production",
        min_replicas: int = 2,
        max_replicas: int = 10,
        target_cpu: int = 70
    ) -> Dict[str, Any]:
        """Create Horizontal Pod Autoscaler manifest."""
        return {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": f"model-{model_info['name']}-hpa",
                "namespace": namespace
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": f"model-{model_info['name']}-v{model_info['version']}"
                },
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": target_cpu
                            }
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": 80
                            }
                        }
                    }
                ],
                "behavior": {
                    "scaleDown": {
                        "stabilizationWindowSeconds": 300,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 50,
                                "periodSeconds": 60
                            }
                        ]
                    },
                    "scaleUp": {
                        "stabilizationWindowSeconds": 60,
                        "policies": [
                            {
                                "type": "Percent",
                                "value": 100,
                                "periodSeconds": 30
                            }
                        ]
                    }
                }
            }
        }
    
    def deploy_to_kubernetes(
        self,
        manifest: Dict[str, Any],
        namespace: str = "production"
    ) -> bool:
        """Deploy manifest to Kubernetes cluster."""
        try:
            # Ensure namespace exists
            try:
                self.k8s_core.read_namespace(namespace)
            except:
                namespace_manifest = client.V1Namespace(
                    metadata=client.V1ObjectMeta(name=namespace)
                )
                self.k8s_core.create_namespace(namespace_manifest)
                logger.info(f"Created namespace: {namespace}")
            
            # Deploy based on manifest kind
            kind = manifest.get("kind")
            name = manifest["metadata"]["name"]
            
            if kind == "Deployment":
                try:
                    # Try to update existing deployment
                    self.k8s_apps.patch_namespaced_deployment(
                        name=name,
                        namespace=namespace,
                        body=manifest
                    )
                    logger.info(f"Updated deployment: {name}")
                except:
                    # Create new deployment
                    self.k8s_apps.create_namespaced_deployment(
                        namespace=namespace,
                        body=manifest
                    )
                    logger.info(f"Created deployment: {name}")
            
            elif kind == "Service":
                try:
                    # Try to update existing service
                    self.k8s_core.patch_namespaced_service(
                        name=name,
                        namespace=namespace,
                        body=manifest
                    )
                    logger.info(f"Updated service: {name}")
                except:
                    # Create new service
                    self.k8s_core.create_namespaced_service(
                        namespace=namespace,
                        body=manifest
                    )
                    logger.info(f"Created service: {name}")
            
            elif kind == "HorizontalPodAutoscaler":
                try:
                    # Try to update existing HPA
                    self.k8s_apps.patch_namespaced_horizontal_pod_autoscaler(
                        name=name,
                        namespace=namespace,
                        body=manifest
                    )
                    logger.info(f"Updated HPA: {name}")
                except:
                    # Create new HPA
                    self.k8s_apps.create_namespaced_horizontal_pod_autoscaler(
                        namespace=namespace,
                        body=manifest
                    )
                    logger.info(f"Created HPA: {name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            return False
    
    def wait_for_deployment(
        self,
        deployment_name: str,
        namespace: str = "production",
        timeout: int = 300
    ) -> bool:
        """Wait for deployment to be ready."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                deployment = self.k8s_apps.read_namespaced_deployment(
                    name=deployment_name,
                    namespace=namespace
                )
                
                # Check if deployment is ready
                if deployment.status.ready_replicas == deployment.spec.replicas:
                    logger.info(f"Deployment {deployment_name} is ready")
                    return True
                
                logger.info(f"Waiting for deployment... ({deployment.status.ready_replicas}/{deployment.spec.replicas} ready)")
                
            except Exception as e:
                logger.error(f"Error checking deployment status: {e}")
            
            time.sleep(10)
        
        logger.error(f"Deployment {deployment_name} timeout after {timeout} seconds")
        return False
    
    def health_check(
        self,
        service_name: str,
        namespace: str = "production",
        endpoint: str = "/health"
    ) -> bool:
        """Perform health check on deployed service."""
        try:
            # Get service details
            service = self.k8s_core.read_namespaced_service(
                name=service_name,
                namespace=namespace
            )
            
            # For in-cluster health check
            service_url = f"http://{service_name}.{namespace}.svc.cluster.local{endpoint}"
            
            response = requests.get(service_url, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"Health check passed for {service_name}")
                return True
            else:
                logger.error(f"Health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return False
    
    def rollback_deployment(
        self,
        deployment_name: str,
        namespace: str = "production"
    ) -> bool:
        """Rollback deployment to previous version."""
        try:
            # Get deployment
            deployment = self.k8s_apps.read_namespaced_deployment(
                name=deployment_name,
                namespace=namespace
            )
            
            # Rollback to previous revision
            deployment.spec.template.metadata.annotations = {
                "kubectl.kubernetes.io/restartedAt": datetime.now().isoformat()
            }
            
            self.k8s_apps.patch_namespaced_deployment(
                name=deployment_name,
                namespace=namespace,
                body=deployment
            )
            
            logger.info(f"Initiated rollback for {deployment_name}")
            return True
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            return False
    
    def deploy_model(
        self,
        model_name: str,
        stage: str = "Production",
        namespace: str = "production",
        strategy: str = "canary",
        replicas: int = 3
    ) -> bool:
        """Complete model deployment pipeline."""
        # Get latest model version
        model_info = self.get_latest_model_version(model_name, stage)
        
        if not model_info:
            logger.error(f"No model found for {model_name} in {stage}")
            return False
        
        logger.info(f"Deploying model {model_name} version {model_info['version']}")
        
        # Create manifests
        deployment_manifest = self.create_deployment_manifest(
            model_info, namespace, replicas, strategy
        )
        service_manifest = self.create_service_manifest(model_info, namespace)
        hpa_manifest = self.create_hpa_manifest(model_info, namespace)
        
        # Save manifests
        os.makedirs("deployment", exist_ok=True)
        
        with open("deployment/deployment.yaml", 'w') as f:
            yaml.dump(deployment_manifest, f)
        
        with open("deployment/service.yaml", 'w') as f:
            yaml.dump(service_manifest, f)
        
        with open("deployment/hpa.yaml", 'w') as f:
            yaml.dump(hpa_manifest, f)
        
        # Deploy to Kubernetes
        success = True
        
        # Deploy service first
        if not self.deploy_to_kubernetes(service_manifest, namespace):
            success = False
        
        # Deploy deployment
        if success and not self.deploy_to_kubernetes(deployment_manifest, namespace):
            success = False
        
        # Deploy HPA
        if success and not self.deploy_to_kubernetes(hpa_manifest, namespace):
            logger.warning("HPA deployment failed, continuing without autoscaling")
        
        # Wait for deployment to be ready
        if success:
            deployment_name = deployment_manifest["metadata"]["name"]
            success = self.wait_for_deployment(deployment_name, namespace)
        
        # Perform health check
        if success:
            service_name = service_manifest["metadata"]["name"]
            success = self.health_check(service_name, namespace)
        
        # Rollback if failed
        if not success:
            logger.error("Deployment failed, initiating rollback")
            self.rollback_deployment(deployment_manifest["metadata"]["name"], namespace)
        
        return success


def main():
    """Main entry point for deployment pipeline."""
    parser = argparse.ArgumentParser(description="Deploy ML model to Kubernetes")
    parser.add_argument("--model-name", required=True, help="Model name in registry")
    parser.add_argument("--stage", default="Production", help="Model stage")
    parser.add_argument("--namespace", default="production", help="Kubernetes namespace")
    parser.add_argument("--strategy", default="canary", choices=["canary", "blue-green", "rolling"])
    parser.add_argument("--replicas", type=int, default=3, help="Number of replicas")
    parser.add_argument("--kubeconfig", help="Path to kubeconfig file")
    
    args = parser.parse_args()
    
    # Initialize deployer
    deployer = ModelDeployer(kubeconfig_path=args.kubeconfig)
    
    # Deploy model
    success = deployer.deploy_model(
        model_name=args.model_name,
        stage=args.stage,
        namespace=args.namespace,
        strategy=args.strategy,
        replicas=args.replicas
    )
    
    if success:
        logger.info("Deployment successful!")
    else:
        logger.error("Deployment failed!")
        exit(1)


if __name__ == "__main__":
    main()