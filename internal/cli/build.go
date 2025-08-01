package cli

import (
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/spf13/cobra"
	"github.com/spf13/viper"
)

var buildCmd = &cobra.Command{
	Use:   "build [target]",
	Short: "Build container images",
	Long: `Build container images for different targets using nerdctl.
	
Available targets:
  runtime     - Base runtime image (default)
  serving     - Model serving image  
  development - Development environment image
  all         - Build all targets`,
	Args: cobra.MaximumNArgs(1),
	RunE: runBuild,
}

var (
	buildRegistry   string
	buildOrg        string
	buildImageName  string
	buildVersion    string
	buildPush       bool
	buildNoCache    bool
	buildTarget     string
)

func init() {
	rootCmd.AddCommand(buildCmd)

	buildCmd.Flags().StringVar(&buildRegistry, "registry", "ghcr.io", "Container registry")
	buildCmd.Flags().StringVar(&buildOrg, "org", "flowops", "Organization name")
	buildCmd.Flags().StringVar(&buildImageName, "image", "mlops-model", "Image name")
	buildCmd.Flags().StringVar(&buildVersion, "version", "", "Image version (default: git commit)")
	buildCmd.Flags().BoolVar(&buildPush, "push", false, "Push image after build")
	buildCmd.Flags().BoolVar(&buildNoCache, "no-cache", false, "Build without cache")
	buildCmd.Flags().StringVar(&buildTarget, "target", "runtime", "Build target")

	// Bind flags to viper
	viper.BindPFlag("build.registry", buildCmd.Flags().Lookup("registry"))
	viper.BindPFlag("build.org", buildCmd.Flags().Lookup("org"))
	viper.BindPFlag("build.image", buildCmd.Flags().Lookup("image"))
	viper.BindPFlag("build.push", buildCmd.Flags().Lookup("push"))
}

func runBuild(cmd *cobra.Command, args []string) error {
	target := buildTarget
	if len(args) > 0 {
		target = args[0]
	}

	// Validate target
	validTargets := []string{"runtime", "serving", "development", "all"}
	if !contains(validTargets, target) {
		return fmt.Errorf("invalid target '%s'. Valid targets: %s", target, strings.Join(validTargets, ", "))
	}

	// Get version from git if not specified
	version := buildVersion
	if version == "" {
		var err error
		version, err = getGitCommit()
		if err != nil {
			version = "latest"
		}
	}

	if verbose {
		fmt.Printf("Building target: %s\n", target)
		fmt.Printf("Registry: %s\n", buildRegistry)
		fmt.Printf("Version: %s\n", version)
	}

	// Check if nerdctl is available
	if err := checkNerdctl(); err != nil {
		return err
	}

	if target == "all" {
		targets := []string{"runtime", "serving", "development"}
		for _, t := range targets {
			if err := buildImage(t, version); err != nil {
				return fmt.Errorf("failed to build %s: %w", t, err)
			}
		}
	} else {
		if err := buildImage(target, version); err != nil {
			return err
		}
	}

	fmt.Printf("✅ Build complete for target: %s\n", target)
	return nil
}

func buildImage(target, version string) error {
	tagSuffix := ""
	switch target {
	case "serving":
		tagSuffix = "-serving"
	case "development":
		tagSuffix = "-dev"
	}

	fullTag := fmt.Sprintf("%s/%s/%s:%s%s", buildRegistry, buildOrg, buildImageName, version, tagSuffix)
	latestTag := fmt.Sprintf("%s/%s/%s:latest%s", buildRegistry, buildOrg, buildImageName, tagSuffix)

	fmt.Printf("🔨 Building %s as %s...\n", target, fullTag)

	args := []string{
		"build",
		"--target", target,
		"--tag", fullTag,
		"--tag", latestTag,
		"--file", "Containerfile",
	}

	if buildNoCache {
		args = append(args, "--no-cache")
	}

	// Add cache args for better performance
	cacheRef := fmt.Sprintf("%s/%s/%s:cache-%s", buildRegistry, buildOrg, buildImageName, target)
	args = append(args,
		"--cache-from", "type=registry,ref="+cacheRef,
		"--cache-to", "type=registry,ref="+cacheRef+",mode=max",
	)

	args = append(args, ".")

	cmd := exec.Command("nerdctl", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("build failed: %w", err)
	}

	// Push if requested
	if buildPush {
		if err := pushImage(fullTag); err != nil {
			return err
		}
		if err := pushImage(latestTag); err != nil {
			return err
		}
	}

	return nil
}

func pushImage(tag string) error {
	fmt.Printf("📤 Pushing %s...\n", tag)
	
	cmd := exec.Command("nerdctl", "push", tag)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("push failed: %w", err)
	}

	return nil
}

func checkNerdctl() error {
	cmd := exec.Command("nerdctl", "version")
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("nerdctl not found or not working. Please install nerdctl: https://github.com/containerd/nerdctl/releases")
	}
	return nil
}

func getGitCommit() (string, error) {
	cmd := exec.Command("git", "rev-parse", "--short", "HEAD")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

func fileExists(filename string) bool {
	info, err := os.Stat(filename)
	if os.IsNotExist(err) {
		return false
	}
	return !info.IsDir()
}