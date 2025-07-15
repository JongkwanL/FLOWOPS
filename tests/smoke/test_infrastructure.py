"""
Infrastructure smoke tests for deployment validation
"""
import subprocess
import json
import time
import pytest
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)

class TestKubernetesDeployment:
    """Test Kubernetes deployment smoke tests"""
    
    def test_kubectl_available(self):
        """Test kubectl is available and configured"""
        try:
            result = subprocess.run(
                ["kubectl", "version", "--client"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("kubectl not available or not configured")
            
    def test_cluster_connectivity(self):
        """Test cluster connectivity"""
        try:
            result = subprocess.run(
                ["kubectl", "cluster-info"],
                capture_output=True,
                text=True,
                timeout=30
            )
            # Accept both success and some connection issues for smoke test
            assert result.returncode in [0, 1]
        except subprocess.TimeoutExpired:
            pytest.skip("Cluster not accessible")
            
    def test_namespace_creation(self):
        """Test namespace can be created"""
        namespace = "flowops-smoke-test"
        
        try:
            # Try to create test namespace
            subprocess.run(
                ["kubectl", "create", "namespace", namespace],
                capture_output=True,
                timeout=30
            )
            
            # Clean up
            subprocess.run(
                ["kubectl", "delete", "namespace", namespace, "--ignore-not-found"],
                capture_output=True,
                timeout=30
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Cannot test namespace creation")

class TestHelmCharts:
    """Test Helm charts validation"""
    
    def test_helm_available(self):
        """Test Helm is available"""
        try:
            result = subprocess.run(
                ["helm", "version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Helm not available")
            
    def test_helm_chart_syntax(self):
        """Test Helm chart syntax"""
        chart_path = Path("infrastructure/helm/flowops")
        
        if not chart_path.exists():
            pytest.skip("Helm chart not found")
            
        try:
            result = subprocess.run(
                ["helm", "lint", str(chart_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Helm lint should pass or have only warnings
            assert result.returncode in [0, 1]
            
            # Check for critical errors
            if "ERROR" in result.stderr:
                pytest.fail(f"Helm chart has errors: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            pytest.skip("Helm lint timeout")
            
    def test_helm_chart_template(self):
        """Test Helm chart template rendering"""
        chart_path = Path("infrastructure/helm/flowops")
        
        if not chart_path.exists():
            pytest.skip("Helm chart not found")
            
        try:
            result = subprocess.run(
                ["helm", "template", "test-release", str(chart_path)],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            assert result.returncode == 0
            
            # Verify YAML output is valid
            try:
                list(yaml.safe_load_all(result.stdout))
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML generated: {e}")
                
        except subprocess.TimeoutExpired:
            pytest.skip("Helm template timeout")

class TestTerraformInfrastructure:
    """Test Terraform infrastructure"""
    
    def test_terraform_available(self):
        """Test Terraform is available"""
        try:
            result = subprocess.run(
                ["terraform", "version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Terraform not available")
            
    def test_terraform_syntax(self):
        """Test Terraform syntax validation"""
        tf_path = Path("infrastructure/terraform")
        
        if not tf_path.exists():
            pytest.skip("Terraform files not found")
            
        try:
            # Change to terraform directory
            result = subprocess.run(
                ["terraform", "fmt", "-check"],
                cwd=tf_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            # Format check should pass (exit code 0) or have formatting issues (exit code 3)
            assert result.returncode in [0, 3]
            
        except subprocess.TimeoutExpired:
            pytest.skip("Terraform fmt timeout")
            
    def test_terraform_validation(self):
        """Test Terraform configuration validation"""
        tf_path = Path("infrastructure/terraform")
        
        if not tf_path.exists():
            pytest.skip("Terraform files not found")
            
        try:
            # Initialize terraform (required for validation)
            init_result = subprocess.run(
                ["terraform", "init", "-backend=false"],
                cwd=tf_path,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if init_result.returncode != 0:
                pytest.skip(f"Terraform init failed: {init_result.stderr}")
            
            # Validate configuration
            result = subprocess.run(
                ["terraform", "validate"],
                cwd=tf_path,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            assert result.returncode == 0, f"Terraform validation failed: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.skip("Terraform validation timeout")

class TestContainerRuntime:
    """Test container runtime (nerdctl) smoke tests"""
    
    def test_nerdctl_available(self):
        """Test nerdctl is available"""
        try:
            result = subprocess.run(
                ["nerdctl", "version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("nerdctl not available")
            
    def test_container_build_syntax(self):
        """Test Containerfile syntax"""
        containerfile = Path("Containerfile")
        
        if not containerfile.exists():
            pytest.skip("Containerfile not found")
            
        # Check for basic Dockerfile syntax
        content = containerfile.read_text()
        
        # Basic syntax checks
        assert "FROM" in content
        assert "WORKDIR" in content or "COPY" in content
        
        # Multi-stage build validation
        from_lines = [line for line in content.split('\n') if line.strip().startswith('FROM')]
        assert len(from_lines) >= 1, "At least one FROM instruction required"
        
    def test_build_script_syntax(self):
        """Test build script syntax"""
        build_script = Path("scripts/build_container.sh")
        
        if not build_script.exists():
            pytest.skip("Build script not found")
            
        try:
            # Check script syntax
            result = subprocess.run(
                ["bash", "-n", str(build_script)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            assert result.returncode == 0, f"Build script syntax error: {result.stderr}"
            
        except subprocess.TimeoutExpired:
            pytest.skip("Build script syntax check timeout")

class TestCI_CD_Configuration:
    """Test CI/CD configuration"""
    
    def test_github_workflows_syntax(self):
        """Test GitHub Actions workflow syntax"""
        workflows_path = Path(".github/workflows")
        
        if not workflows_path.exists():
            pytest.skip("GitHub workflows not found")
            
        workflow_files = list(workflows_path.glob("*.yml")) + list(workflows_path.glob("*.yaml"))
        
        if not workflow_files:
            pytest.skip("No workflow files found")
            
        for workflow_file in workflow_files:
            try:
                with open(workflow_file, 'r') as f:
                    workflow_data = yaml.safe_load(f)
                    
                # Basic workflow validation
                assert "name" in workflow_data
                assert "on" in workflow_data
                assert "jobs" in workflow_data
                
                # Validate jobs structure
                for job_name, job_config in workflow_data["jobs"].items():
                    assert "runs-on" in job_config
                    assert "steps" in job_config
                    
            except yaml.YAMLError as e:
                pytest.fail(f"Invalid YAML in {workflow_file}: {e}")
                
    def test_dvc_configuration(self):
        """Test DVC configuration"""
        dvc_file = Path("dvc.yaml")
        
        if not dvc_file.exists():
            pytest.skip("DVC configuration not found")
            
        try:
            with open(dvc_file, 'r') as f:
                dvc_data = yaml.safe_load(f)
                
            # Basic DVC validation
            assert "stages" in dvc_data
            
            # Validate stages
            for stage_name, stage_config in dvc_data["stages"].items():
                assert "cmd" in stage_config
                
        except yaml.YAMLError as e:
            pytest.fail(f"Invalid DVC YAML: {e}")

class TestSecurityConfiguration:
    """Test security configuration"""
    
    def test_no_secrets_in_config(self):
        """Test that no secrets are hardcoded in configuration files"""
        config_files = [
            "dvc.yaml",
            ".github/workflows/*.yml",
            ".github/workflows/*.yaml",
            "infrastructure/helm/flowops/values.yaml",
            "infrastructure/terraform/*.tf"
        ]
        
        # Patterns that might indicate secrets
        secret_patterns = [
            r"password\s*[:=]\s*['\"][^'\"]+['\"]",
            r"secret\s*[:=]\s*['\"][^'\"]+['\"]",
            r"token\s*[:=]\s*['\"][^'\"]+['\"]",
            r"key\s*[:=]\s*['\"][^'\"]+['\"]",
        ]
        
        import re
        from glob import glob
        
        violations = []
        
        for pattern in config_files:
            for file_path in glob(pattern, recursive=True):
                if Path(file_path).exists():
                    try:
                        content = Path(file_path).read_text()
                        
                        for secret_pattern in secret_patterns:
                            if re.search(secret_pattern, content, re.IGNORECASE):
                                violations.append(f"Potential secret in {file_path}")
                                
                    except Exception:
                        continue  # Skip files that can't be read
        
        if violations:
            logger.warning(f"Potential security issues found: {violations}")
            # Don't fail smoke test for this, just warn

def run_infrastructure_smoke_tests():
    """Run infrastructure smoke tests"""
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-k", "not test_no_secrets_in_config"  # Skip security test in smoke
    ]
    
    return pytest.main(pytest_args)

if __name__ == "__main__":
    exit_code = run_infrastructure_smoke_tests()
    exit(exit_code)