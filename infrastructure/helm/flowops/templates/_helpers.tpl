{{/*
Expand the name of the chart.
*/}}
{{- define "flowops.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "flowops.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "flowops.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "flowops.labels" -}}
helm.sh/chart: {{ include "flowops.chart" . }}
{{ include "flowops.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "flowops.selectorLabels" -}}
app.kubernetes.io/name: {{ include "flowops.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use for model serving
*/}}
{{- define "flowops.serviceAccountName.modelServing" -}}
{{- if .Values.serviceAccount.modelServing.create }}
{{- default (printf "%s-model-serving" (include "flowops.fullname" .)) .Values.serviceAccount.modelServing.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.modelServing.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the service account to use for MLflow
*/}}
{{- define "flowops.serviceAccountName.mlflow" -}}
{{- if .Values.serviceAccount.mlflow.create }}
{{- default (printf "%s-mlflow" (include "flowops.fullname" .)) .Values.serviceAccount.mlflow.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.mlflow.name }}
{{- end }}
{{- end }}

{{/*
MLflow database URL
*/}}
{{- define "flowops.mlflow.databaseUrl" -}}
{{- if .Values.postgresql.enabled }}
postgresql://{{ .Values.postgresql.auth.username }}:{{ .Values.postgresql.auth.password }}@{{ include "flowops.fullname" . }}-postgresql:5432/{{ .Values.postgresql.auth.database }}
{{- else }}
{{- .Values.mlflow.trackingServer.database.url }}
{{- end }}
{{- end }}

{{/*
S3 artifacts URL
*/}}
{{- define "flowops.mlflow.artifactsUrl" -}}
{{- if .Values.mlflow.s3.enabled }}
s3://{{ .Values.mlflow.s3.bucket }}
{{- else }}
file:///mlflow/artifacts
{{- end }}
{{- end }}

{{/*
Common environment variables for MLflow
*/}}
{{- define "flowops.mlflow.env" -}}
- name: MLFLOW_BACKEND_STORE_URI
  value: {{ include "flowops.mlflow.databaseUrl" . | quote }}
- name: MLFLOW_DEFAULT_ARTIFACT_ROOT
  value: {{ include "flowops.mlflow.artifactsUrl" . | quote }}
{{- if .Values.mlflow.s3.enabled }}
- name: AWS_DEFAULT_REGION
  value: {{ .Values.mlflow.s3.region | quote }}
{{- end }}
{{- end }}

{{/*
Generate certificates for MLflow if needed
*/}}
{{- define "flowops.gen-certs" -}}
{{- $altNames := list ( printf "%s.%s" (include "flowops.name" .) .Release.Namespace ) ( printf "%s.%s.svc" (include "flowops.name" .) .Release.Namespace ) -}}
{{- $ca := genCA "flowops-ca" 365 -}}
{{- $cert := genSignedCert ( include "flowops.name" . ) nil $altNames 365 $ca -}}
tls.crt: {{ $cert.Cert | b64enc }}
tls.key: {{ $cert.Key | b64enc }}
{{- end }}

{{/*
Argo Rollouts strategy
*/}}
{{- define "flowops.rolloutStrategy" -}}
{{- if eq .Values.argoRollouts.strategy "canary" }}
canary:
  maxSurge: 25%
  maxUnavailable: 0
  steps:
  - setWeight: 10
  - pause: {duration: 1m}
  - setWeight: 20
  - pause: {duration: 2m}
  - setWeight: 50
  - pause: {duration: 5m}
  - setWeight: 80
  - pause: {duration: 5m}
{{- else if eq .Values.argoRollouts.strategy "blueGreen" }}
blueGreen:
  autoPromotionEnabled: false
  scaleDownDelaySeconds: 30
  prePromotionAnalysis:
    templates:
    - templateName: success-rate
    args:
    - name: service-name
      value: {{ include "flowops.fullname" . }}-model-serving
  postPromotionAnalysis:
    templates:
    - templateName: success-rate
    args:
    - name: service-name
      value: {{ include "flowops.fullname" . }}-model-serving
{{- end }}
{{- end }}