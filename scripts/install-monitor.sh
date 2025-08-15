#!/bin/bash
# Installation script for FlowOps monitoring agent

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Configuration
FLOWOPS_USER="flowops"
FLOWOPS_GROUP="flowops"
INSTALL_DIR="/usr/local/bin"
CONFIG_DIR="/etc/flowops"
LOG_DIR="/var/log/flowops"
DATA_DIR="/var/lib/flowops"
SERVICE_FILE="monitor-agent.service"

# Check if running as root
if [[ $EUID -ne 0 ]]; then
   print_message "$RED" "This script must be run as root (use sudo)"
   exit 1
fi

print_message "$GREEN" "=== FlowOps Monitoring Agent Installation ==="

# Create user and group
if ! id "$FLOWOPS_USER" &>/dev/null; then
    print_message "$YELLOW" "Creating user $FLOWOPS_USER..."
    useradd --system --home-dir "$DATA_DIR" --shell /bin/false "$FLOWOPS_USER"
else
    print_message "$GREEN" "User $FLOWOPS_USER already exists"
fi

# Create directories
print_message "$YELLOW" "Creating directories..."
mkdir -p "$CONFIG_DIR" "$LOG_DIR" "$DATA_DIR"
chown "$FLOWOPS_USER:$FLOWOPS_GROUP" "$LOG_DIR" "$DATA_DIR"
chmod 755 "$CONFIG_DIR"
chmod 750 "$LOG_DIR" "$DATA_DIR"

# Install binary
if [[ -f "bin/flowops" ]]; then
    print_message "$YELLOW" "Installing FlowOps binary..."
    cp bin/flowops "$INSTALL_DIR/"
    chmod 755 "$INSTALL_DIR/flowops"
    chown root:root "$INSTALL_DIR/flowops"
else
    print_message "$RED" "Error: bin/flowops not found. Please build first with 'make build'"
    exit 1
fi

# Install configuration
if [[ -f "configs/monitor-config.yaml" ]]; then
    print_message "$YELLOW" "Installing configuration..."
    cp configs/monitor-config.yaml "$CONFIG_DIR/"
    chown root:"$FLOWOPS_GROUP" "$CONFIG_DIR/monitor-config.yaml"
    chmod 640 "$CONFIG_DIR/monitor-config.yaml"
else
    print_message "$YELLOW" "Warning: No config file found, using defaults"
fi

# Install systemd service
if [[ -f "scripts/$SERVICE_FILE" ]]; then
    print_message "$YELLOW" "Installing systemd service..."
    cp "scripts/$SERVICE_FILE" "/etc/systemd/system/"
    systemctl daemon-reload
    
    print_message "$GREEN" "Service installed. Enable with:"
    print_message "$GREEN" "  sudo systemctl enable monitor-agent"
    print_message "$GREEN" "  sudo systemctl start monitor-agent"
else
    print_message "$YELLOW" "Warning: Service file not found"
fi

# Create logrotate configuration
print_message "$YELLOW" "Setting up log rotation..."
cat > /etc/logrotate.d/flowops-monitor << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 $FLOWOPS_USER $FLOWOPS_GROUP
    postrotate
        systemctl reload monitor-agent > /dev/null 2>&1 || true
    endscript
}
EOF

# Create monitoring directories
mkdir -p "$DATA_DIR/metrics" "$DATA_DIR/status"
chown "$FLOWOPS_USER:$FLOWOPS_GROUP" "$DATA_DIR/metrics" "$DATA_DIR/status"

# Set up firewall (if ufw is available)
if command -v ufw &> /dev/null; then
    print_message "$YELLOW" "Configuring firewall..."
    ufw allow 9090/tcp comment "FlowOps Monitoring Agent"
fi

print_message "$GREEN" "✅ Installation completed successfully!"
print_message "$GREEN" ""
print_message "$GREEN" "Next steps:"
print_message "$GREEN" "1. Review configuration: $CONFIG_DIR/monitor-config.yaml"
print_message "$GREEN" "2. Enable service: sudo systemctl enable monitor-agent"
print_message "$GREEN" "3. Start service: sudo systemctl start monitor-agent"
print_message "$GREEN" "4. Check status: sudo systemctl status monitor-agent"
print_message "$GREEN" "5. View logs: sudo journalctl -u monitor-agent -f"
print_message "$GREEN" "6. Access metrics: http://localhost:9090/metrics"
print_message "$GREEN" ""
print_message "$YELLOW" "For manual testing: $INSTALL_DIR/flowops agent --help"