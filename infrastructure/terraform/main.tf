terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.24"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.12"
    }
  }
  
  backend "s3" {
    bucket         = "flowops-terraform-state"
    key            = "eks/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "flowops-terraform-locks"
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "FlowOps"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CreatedAt   = timestamp()
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# VPC Module for EKS
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"
  
  name = "${var.cluster_name}-vpc"
  cidr = var.vpc_cidr
  
  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnet_cidrs
  public_subnets  = var.public_subnet_cidrs
  
  enable_nat_gateway   = true
  single_nat_gateway   = var.environment != "production"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  # VPC Flow Logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_iam_role  = true
  create_flow_log_cloudwatch_log_group = true
  
  public_subnet_tags = {
    "kubernetes.io/role/elb"                    = 1
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
  
  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"           = 1
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
  
  tags = {
    "kubernetes.io/cluster/${var.cluster_name}" = "shared"
  }
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 20.0"
  
  cluster_name    = var.cluster_name
  cluster_version = var.kubernetes_version
  
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true
  
  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }
  
  # Cluster Addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
      configuration_values = jsonencode({
        env = {
          ENABLE_PREFIX_DELEGATION = "true"
          WARM_PREFIX_TARGET       = "1"
        }
      })
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }
  
  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets
  
  # Security Group Rules
  cluster_security_group_additional_rules = {
    ingress_nodes_ephemeral_ports_tcp = {
      description                = "Nodes on ephemeral ports"
      protocol                   = "tcp"
      from_port                  = 1025
      to_port                    = 65535
      type                       = "ingress"
      source_node_security_group = true
    }
  }
  
  # Node Security Group Rules
  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node to node all ports/protocols"
      protocol    = "-1"
      from_port   = 0
      to_port     = 0
      type        = "ingress"
      self        = true
    }
    
    ingress_cluster_all = {
      description                   = "Cluster to node all ports/protocols"
      protocol                      = "-1"
      from_port                     = 0
      to_port                       = 0
      type                          = "ingress"
      source_cluster_security_group = true
    }
  }
  
  # Managed Node Groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      name            = "${var.cluster_name}-general"
      use_name_prefix = true
      
      subnet_ids = module.vpc.private_subnets
      
      min_size     = var.node_group_min_size
      max_size     = var.node_group_max_size
      desired_size = var.node_group_desired_size
      
      instance_types = var.node_instance_types
      capacity_type  = var.capacity_type
      
      # Use latest AMI
      ami_type = "AL2_x86_64"
      
      # Enable detailed monitoring
      enable_monitoring = true
      
      # Block device mappings
      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = 100
            volume_type           = "gp3"
            iops                  = 3000
            throughput            = 125
            encrypted             = true
            kms_key_id            = aws_kms_key.ebs.arn
            delete_on_termination = true
          }
        }
      }
      
      labels = {
        Environment = var.environment
        NodeType    = "general"
      }
      
      tags = {
        "k8s.io/cluster-autoscaler/enabled"             = "true"
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
      }
    }
    
    # GPU nodes for ML workloads
    gpu = {
      name            = "${var.cluster_name}-gpu"
      use_name_prefix = true
      
      subnet_ids = module.vpc.private_subnets
      
      min_size     = var.gpu_node_min_size
      max_size     = var.gpu_node_max_size
      desired_size = var.gpu_node_desired_size
      
      instance_types = var.gpu_instance_types
      capacity_type  = var.gpu_capacity_type
      
      # GPU AMI
      ami_type = "AL2_x86_64_GPU"
      
      # Taints for GPU nodes
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      labels = {
        Environment             = var.environment
        NodeType                = "gpu"
        "nvidia.com/gpu"        = "true"
        "k8s.amazonaws.com/accelerator" = "nvidia-tesla"
      }
      
      tags = {
        "k8s.io/cluster-autoscaler/enabled"             = "true"
        "k8s.io/cluster-autoscaler/${var.cluster_name}" = "owned"
        "k8s.io/cluster-autoscaler/node-template/label/nvidia.com/gpu" = "true"
        "k8s.io/cluster-autoscaler/node-template/taint/nvidia.com/gpu" = "true:NoSchedule"
      }
    }
  }
  
  # OIDC Provider for IRSA
  enable_irsa = true
  
  # CloudWatch Logs
  cluster_enabled_log_types = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  
  tags = {
    Environment = var.environment
    GithubRepo  = "flowops"
  }
}

# KMS Key for EKS cluster encryption
resource "aws_kms_key" "eks" {
  description             = "EKS cluster encryption key"
  deletion_window_in_days = 10
  enable_key_rotation     = true
  
  tags = {
    Name = "${var.cluster_name}-eks-key"
  }
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${var.cluster_name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# KMS Key for EBS encryption
resource "aws_kms_key" "ebs" {
  description             = "EBS volume encryption key"
  deletion_window_in_days = 10
  enable_key_rotation     = true
  
  tags = {
    Name = "${var.cluster_name}-ebs-key"
  }
}

resource "aws_kms_alias" "ebs" {
  name          = "alias/${var.cluster_name}-ebs"
  target_key_id = aws_kms_key.ebs.key_id
}

# S3 Buckets for MLflow and DVC
resource "aws_s3_bucket" "mlflow" {
  bucket = "${var.cluster_name}-mlflow-artifacts"
  
  tags = {
    Name = "${var.cluster_name}-mlflow-artifacts"
    Type = "MLflow"
  }
}

resource "aws_s3_bucket_versioning" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "dvc" {
  bucket = "${var.cluster_name}-dvc-storage"
  
  tags = {
    Name = "${var.cluster_name}-dvc-storage"
    Type = "DVC"
  }
}

resource "aws_s3_bucket_versioning" "dvc" {
  bucket = aws_s3_bucket.dvc.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "dvc" {
  bucket = aws_s3_bucket.dvc.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# RDS for MLflow backend store
resource "aws_db_subnet_group" "mlflow" {
  name       = "${var.cluster_name}-mlflow"
  subnet_ids = module.vpc.private_subnets
  
  tags = {
    Name = "${var.cluster_name}-mlflow-db-subnet"
  }
}

resource "aws_security_group" "mlflow_db" {
  name_prefix = "${var.cluster_name}-mlflow-db"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name = "${var.cluster_name}-mlflow-db-sg"
  }
}

resource "aws_db_instance" "mlflow" {
  identifier     = "${var.cluster_name}-mlflow"
  engine         = "postgres"
  engine_version = "15.4"
  instance_class = var.db_instance_class
  
  allocated_storage     = 100
  storage_type          = "gp3"
  storage_encrypted     = true
  kms_key_id            = aws_kms_key.ebs.arn
  
  db_name  = "mlflow"
  username = "mlflow"
  password = var.db_password  # Use AWS Secrets Manager in production
  
  vpc_security_group_ids = [aws_security_group.mlflow_db.id]
  db_subnet_group_name   = aws_db_subnet_group.mlflow.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  deletion_protection = var.environment == "production"
  skip_final_snapshot = var.environment != "production"
  
  enabled_cloudwatch_logs_exports = ["postgresql"]
  
  tags = {
    Name = "${var.cluster_name}-mlflow-db"
  }
}

# IAM Roles for Service Accounts (IRSA)
module "mlflow_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"
  
  role_name = "${var.cluster_name}-mlflow-irsa"
  
  attach_external_secrets_policy = false
  
  role_policy_arns = {
    policy = aws_iam_policy.mlflow.arn
  }
  
  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["mlflow:mlflow-sa"]
    }
  }
}

resource "aws_iam_policy" "mlflow" {
  name_prefix = "${var.cluster_name}-mlflow"
  description = "IAM policy for MLflow service account"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject",
          "s3:ListBucket"
        ]
        Resource = [
          aws_s3_bucket.mlflow.arn,
          "${aws_s3_bucket.mlflow.arn}/*"
        ]
      }
    ]
  })
}

# Cluster Autoscaler IRSA
module "cluster_autoscaler_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"
  
  role_name                        = "${var.cluster_name}-cluster-autoscaler"
  attach_cluster_autoscaler_policy = true
  cluster_autoscaler_cluster_names = [module.eks.cluster_name]
  
  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["kube-system:cluster-autoscaler"]
    }
  }
}

# External Secrets Operator IRSA
module "external_secrets_irsa" {
  source  = "terraform-aws-modules/iam/aws//modules/iam-role-for-service-accounts-eks"
  version = "~> 5.0"
  
  role_name                      = "${var.cluster_name}-external-secrets"
  attach_external_secrets_policy = true
  
  oidc_providers = {
    main = {
      provider_arn               = module.eks.oidc_provider_arn
      namespace_service_accounts = ["external-secrets:external-secrets-sa"]
    }
  }
}

# Configure kubectl and update kubeconfig
resource "null_resource" "update_kubeconfig" {
  provisioner "local-exec" {
    command = "aws eks update-kubeconfig --region ${var.aws_region} --name ${module.eks.cluster_name}"
  }
  
  depends_on = [module.eks]
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_iam_role_arn" {
  description = "IAM role ARN of the EKS cluster"
  value       = module.eks.cluster_iam_role_arn
}

output "mlflow_db_endpoint" {
  description = "MLflow database endpoint"
  value       = aws_db_instance.mlflow.endpoint
  sensitive   = true
}

output "mlflow_bucket" {
  description = "MLflow S3 bucket name"
  value       = aws_s3_bucket.mlflow.id
}

output "dvc_bucket" {
  description = "DVC S3 bucket name"
  value       = aws_s3_bucket.dvc.id
}