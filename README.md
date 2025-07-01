# FlowOps - End-to-End MLOps Automation Platform

## 🎯 프로젝트 개요
FlowOps는 머신러닝 모델의 전체 생명주기를 자동화하는 MLOps 플랫폼입니다. 데이터 버저닝부터 모델 학습, 실험 추적, 자동 배포, 모니터링까지 완전한 ML 파이프라인을 제공합니다.

## 🚀 핵심 기능
- **실험 추적**: MLflow 기반 실험 관리 및 메트릭 추적
- **데이터 버저닝**: DVC를 통한 데이터셋 버전 관리
- **자동화 파이프라인**: GitHub Actions CI/CD 통합
- **스마트 배포**: Canary/Blue-Green 배포 전략
- **모델 레지스트리**: 중앙화된 모델 버전 관리
- **자동 롤백**: 성능 저하 시 자동 롤백 메커니즘

## 🛠️ 기술 스택
### MLOps Core
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **Model Registry**: MLflow Model Registry
- **Pipeline Orchestration**: GitHub Actions, Argo Workflows

### Deployment
- **Package Manager**: Helm
- **Progressive Delivery**: Argo Rollouts
- **Container Registry**: ECR/Harbor
- **GitOps**: ArgoCD

### Infrastructure
- **Platform**: Kubernetes/EKS
- **IaC**: Terraform
- **Secrets**: External Secrets Operator
- **Monitoring**: Prometheus, Grafana

## 📊 성능 목표
- **배포 시간**: < 10분 (commit to production)
- **롤백 시간**: < 2분
- **실험 재현성**: 100%
- **파이프라인 성공률**: > 95%
- **모델 드리프트 감지**: < 24시간

## 🏗️ 아키텍처
```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│   Git Push  │────▶│GitHub Actions│────▶│    Build     │
└─────────────┘     └──────────────┘     │   & Test     │
                                          └──────┬───────┘
                                                 │
                    ┌──────────────┬─────────────▼────────────┬──────────────┐
                    │              │                          │              │
            ┌───────▼──────┐ ┌────▼─────┐         ┌──────────▼──────┐ ┌─────▼──────┐
            │    MLflow    │ │   DVC    │         │  Model Registry │ │   Helm     │
            │   Tracking   │ │  Storage │         │                 │ │   Charts   │
            └──────────────┘ └──────────┘         └─────────────────┘ └─────┬──────┘
                                                                             │
                                                                    ┌────────▼────────┐
                                                                    │  Argo Rollouts  │
                                                                    │  (Deployment)   │
                                                                    └────────┬────────┘
                                                                             │
                                                    ┌────────────────────────▼────────────────────────┐
                                                    │                 Kubernetes                       │
                                                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐     │
                                                    │  │  Canary  │  │  Stable  │  │  Preview │     │
                                                    │  └──────────┘  └──────────┘  └──────────┘     │
                                                    └─────────────────────────────────────────────────┘
```

## 📁 프로젝트 구조
```
FlowOps/
├── .github/
│   └── workflows/     # CI/CD pipelines
├── pipelines/
│   ├── training/      # Training pipelines
│   ├── evaluation/    # Model evaluation
│   └── deployment/    # Deployment pipelines
├── infrastructure/
│   ├── terraform/     # IaC definitions
│   ├── helm/          # Helm charts
│   └── k8s/           # K8s manifests
├── mlflow/
│   ├── experiments/   # Experiment configs
│   └── models/        # Model artifacts
├── dvc/
│   ├── config/        # DVC configuration
│   └── pipelines/     # DVC pipelines
├── monitoring/
│   ├── dashboards/    # Grafana dashboards
│   └── alerts/        # Alert rules
└── tests/
    ├── unit/          # Unit tests
    ├── integration/   # Integration tests
    └── smoke/         # Smoke tests
```

## 🚦 Pipeline Stages
1. **Code Commit** → Trigger pipeline
2. **Validation** → Linting, type checking
3. **Testing** → Unit, integration tests
4. **Build** → Container image creation
5. **Security Scan** → Trivy, Snyk scanning
6. **Model Training** → Experiment tracking
7. **Model Evaluation** → Performance metrics
8. **Model Registry** → Version and store
9. **Deployment** → Progressive rollout
10. **Monitoring** → Performance tracking

## 🔧 설치 및 실행
```bash
# MLflow 서버 시작
mlflow server --backend-store-uri sqlite:///mlflow.db

# DVC 초기화
dvc init
dvc remote add -d storage s3://my-bucket/dvc

# 파이프라인 실행
dvc repro

# Helm 배포
helm install flowops ./infrastructure/helm/flowops

# ArgoCD 애플리케이션 생성
kubectl apply -f infrastructure/k8s/argocd-app.yaml
```

## 📈 개발 로드맵
- [x] Week 7: 데이터 및 모델 버저닝
- [x] Week 8: CI/CD 파이프라인 구축
- [x] Week 9: 배포 자동화
- [ ] A/B 테스팅 프레임워크
- [ ] Feature Store 통합
- [ ] 모델 드리프트 모니터링
- [ ] AutoML 파이프라인

## 🎯 주요 기능
### 실험 관리
- **자동 로깅**: 하이퍼파라미터, 메트릭, 아티팩트
- **실험 비교**: 병렬 실험 결과 비교
- **재현성**: 완벽한 실험 재현 보장

### 모델 배포
- **전략**: Canary, Blue-Green, Rolling
- **자동 롤백**: 메트릭 기반 자동 롤백
- **A/B 테스팅**: 트래픽 분할 테스트

### 모니터링
- **모델 성능**: 실시간 성능 추적
- **데이터 드리프트**: 입력 데이터 변화 감지
- **리소스 사용량**: GPU/CPU/메모리 모니터링

## 📚 문서
- [Pipeline Configuration](./docs/pipelines.md)
- [MLflow Guide](./docs/mlflow.md)
- [Deployment Strategies](./docs/deployment.md)
- [Monitoring Setup](./docs/monitoring.md)

## 🔐 보안
- 이미지 스캔 (Trivy)
- 시크릿 관리 (External Secrets)
- RBAC 정책
- 네트워크 정책

## 📄 라이선스
MIT License