#!/bin/bash

# ============================================================================
# MBTA Agentcy - Windows-Safe Upload Deployment
# ============================================================================
# Fixed for Git Bash on Windows - properly excludes venv/
# ============================================================================

set -e

OPENAI_API_KEY="$1"
MBTA_API_KEY="$2"
LOCAL_PROJECT_PATH="$3"
REGISTRY_URL="${4:-}"
REGION="${5:-us-east}"
INSTANCE_TYPE="${6:-g6-standard-4}"
ROOT_PASSWORD="${7:-}"

if [ -z "$OPENAI_API_KEY" ] || [ -z "$MBTA_API_KEY" ] || [ -z "$LOCAL_PROJECT_PATH" ]; then
    echo "❌ Usage: $0 <OPENAI_KEY> <MBTA_KEY> <LOCAL_PATH> [REGISTRY_URL]"
    exit 1
fi

if [ ! -d "$LOCAL_PROJECT_PATH" ]; then
    echo "❌ Path does not exist: $LOCAL_PROJECT_PATH"
    exit 1
fi

if [ -z "$ROOT_PASSWORD" ]; then
    ROOT_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
fi

FIREWALL_LABEL="mbta-agentcy-firewall"
SSH_KEY_LABEL="mbta-agentcy-key"
IMAGE_ID="linode/ubuntu22.04"
DEPLOYMENT_ID=$(date +%Y%m%d-%H%M%S)

echo "🚇 MBTA Agentcy - Upload Deployment"
echo "Deployment ID: $DEPLOYMENT_ID"
echo ""

# [1/9] Check Linode CLI
echo "[1/9] Checking Linode CLI..."
if ! linode-cli --version >/dev/null 2>&1; then
    echo "❌ Linode CLI not installed"
    exit 1
fi
echo "✅ Linode CLI ready"

# [2/9] Smart packaging - ONLY include what's needed
echo "[2/9] Packaging project..."
cd "$LOCAL_PROJECT_PATH"
TARBALL_NAME="mbta-${DEPLOYMENT_ID}.tar.gz"

# Check for venv and warn
if [ -d "venv" ] || [ -d ".venv" ]; then
    echo "⚠️  Detected venv/ folder - will be excluded"
fi

# Package ONLY the directories we need
echo "   Packaging: src/, docker/, observability/"
tar -czf "/tmp/$TARBALL_NAME" \
    --exclude='venv' \
    --exclude='.venv' \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.env' \
    --exclude='*.log' \
    --exclude='*.egg-info' \
    --exclude='.pytest_cache' \
    --exclude='node_modules' \
    --exclude='*.key' \
    --exclude='*.key.pub' \
    src/ \
    docker/ \
    observability/ \
    docker-compose-observability.yml \
    2>/dev/null || true

# Also include these if they exist
cd "$LOCAL_PROJECT_PATH"
if [ -f "requirements.txt" ]; then
    tar -rzf "/tmp/$TARBALL_NAME" requirements.txt 2>/dev/null || true
fi

TARBALL_SIZE=$(du -h "/tmp/$TARBALL_NAME" | cut -f1)
TARBALL_BYTES=$(du -b "/tmp/$TARBALL_NAME" | cut -f1)

echo "✅ Packaged: $TARBALL_SIZE"

# Verify size is reasonable
if [ "$TARBALL_BYTES" -gt 52428800 ]; then  # 50MB
    echo "❌ ERROR: Tarball is still too large ($TARBALL_SIZE)"
    echo ""
    echo "Something unexpected is included. Check contents:"
    echo "  tar -tzf /tmp/$TARBALL_NAME | head -100"
    echo ""
    echo "Expected size: <10MB for source code only"
    rm "/tmp/$TARBALL_NAME"
    exit 1
fi

# [3/9] Setup firewall
echo "[3/9] Setting up firewall..."
FIREWALL_ID=$(linode-cli firewalls list --text --no-headers --format="id,label" | grep "$FIREWALL_LABEL" | cut -f1 || echo "")

INBOUND_RULES='[
    {"protocol": "TCP", "ports": "22", "addresses": {"ipv4": ["0.0.0.0/0"]}, "action": "ACCEPT"},
    {"protocol": "TCP", "ports": "3000", "addresses": {"ipv4": ["0.0.0.0/0"]}, "action": "ACCEPT"},
    {"protocol": "TCP", "ports": "8100-8003", "addresses": {"ipv4": ["0.0.0.0/0"]}, "action": "ACCEPT"},
    {"protocol": "TCP", "ports": "16686", "addresses": {"ipv4": ["0.0.0.0/0"]}, "action": "ACCEPT"},
    {"protocol": "TCP", "ports": "3001", "addresses": {"ipv4": ["0.0.0.0/0"]}, "action": "ACCEPT"}
]'

if [ -z "$FIREWALL_ID" ]; then
    linode-cli firewalls create \
        --label "$FIREWALL_LABEL" \
        --rules.inbound_policy DROP \
        --rules.outbound_policy ACCEPT \
        --rules.inbound "$INBOUND_RULES" >/dev/null
    FIREWALL_ID=$(linode-cli firewalls list --text --no-headers --format="id,label" | grep "$FIREWALL_LABEL" | cut -f1)
    echo "✅ Created firewall"
else
    echo "✅ Using existing firewall"
fi

# [4/9] Setup SSH key
echo "[4/9] Setting up SSH key..."
if [ ! -f "${SSH_KEY_LABEL}.pub" ]; then
    ssh-keygen -t rsa -b 4096 -f "$SSH_KEY_LABEL" -N "" -C "mbta-$DEPLOYMENT_ID" >/dev/null 2>&1
    echo "✅ Generated SSH key"
else
    echo "✅ Using existing SSH key"
fi

# [5/9] Launch instance
echo "[5/9] Launching Linode..."
INSTANCE_ID=$(linode-cli linodes create \
    --type "$INSTANCE_TYPE" \
    --region "$REGION" \
    --image "$IMAGE_ID" \
    --label "mbta-agentcy-$DEPLOYMENT_ID" \
    --tags "MBTA-Agentcy" \
    --root_pass "$ROOT_PASSWORD" \
    --authorized_keys "$(cat ${SSH_KEY_LABEL}.pub)" \
    --firewall_id "$FIREWALL_ID" \
    --text --no-headers --format="id")
echo "✅ Instance ID: $INSTANCE_ID"

echo "   Waiting for instance..."
while true; do
    STATUS=$(linode-cli linodes view "$INSTANCE_ID" --text --no-headers --format="status")
    [ "$STATUS" = "running" ] && break
    sleep 5
done

PUBLIC_IP=$(linode-cli linodes view "$INSTANCE_ID" --text --no-headers --format="ipv4")
echo "✅ Public IP: $PUBLIC_IP"

# [6/9] Wait for SSH
echo "[6/9] Waiting for SSH..."
for i in {1..60}; do
    if ssh -i "$SSH_KEY_LABEL" -o StrictHostKeyChecking=no -o ConnectTimeout=5 \
        "root@$PUBLIC_IP" "echo ready" >/dev/null 2>&1; then
        echo "✅ SSH ready"
        break
    fi
    [ $i -eq 60 ] && { echo "❌ SSH timeout"; exit 1; }
    sleep 5
done

# [7/9] Upload (should be fast now - <10MB)
echo "[7/9] Uploading project ($TARBALL_SIZE)..."

if scp -i "$SSH_KEY_LABEL" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 \
    "/tmp/$TARBALL_NAME" "root@$PUBLIC_IP:/tmp/"; then
    echo "✅ Upload successful!"
else
    echo "❌ Upload failed"
    echo "🛑 Cleanup: linode-cli linodes delete $INSTANCE_ID"
    exit 1
fi

# [8/9] Install packages
echo "[8/9] Installing system packages (5-10 min)..."
ssh -i "$SSH_KEY_LABEL" -o StrictHostKeyChecking=no "root@$PUBLIC_IP" bash << 'PACKAGES'
set -e
export DEBIAN_FRONTEND=noninteractive

echo "Updating system..."
apt-get update -y >/dev/null 2>&1
apt-get install -y software-properties-common >/dev/null 2>&1

echo "Installing Python 3.11..."
add-apt-repository -y ppa:deadsnakes/ppa >/dev/null 2>&1
apt-get update -y >/dev/null 2>&1
apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip git supervisor >/dev/null 2>&1

echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh >/dev/null 2>&1
systemctl enable docker >/dev/null 2>&1
systemctl start docker >/dev/null 2>&1

echo "✅ System packages installed"
PACKAGES

# [9/9] Setup services
echo "[9/9] Configuring services (5-10 min)..."

ssh -i "$SSH_KEY_LABEL" -o StrictHostKeyChecking=no "root@$PUBLIC_IP" bash << SETUP
set -e

echo "Setting up MBTA Agentcy..."
cd /opt
mkdir -p mbta-agentcy
cd mbta-agentcy

echo "Extracting project..."
tar -xzf /tmp/$TARBALL_NAME
echo "✅ Extracted"

# Create venv with Python 3.11
echo "Creating Python 3.11 virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

python --version

# Install dependencies
echo "Installing Python packages (this takes 5 minutes)..."
pip install --upgrade pip >/dev/null 2>&1

pip install fastapi uvicorn httpx openai scikit-learn numpy pydantic \
    python-dotenv aiofiles websockets mcp langchain-core langgraph \
    opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp >/dev/null 2>&1

echo "Installing mbta-mcp..."
pip install git+https://github.com/cubismod/mbta-mcp.git >/dev/null 2>&1

echo "✅ Python packages installed"

# Create .env
cat > .env << ENV
OPENAI_API_KEY=$OPENAI_API_KEY
MBTA_API_KEY=$MBTA_API_KEY
REGISTRY_URL=$REGISTRY_URL
PUBLIC_IP=$PUBLIC_IP
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
ENV

echo "✅ Environment configured"

# Start observability if docker-compose exists
if [ -f docker-compose-observability.yml ]; then
    echo "Starting observability stack..."
    docker compose -f docker-compose-observability.yml up -d 2>/dev/null || echo "⚠️  Observability skipped"
    sleep 10
fi

# Create supervisor configs
cat > /etc/supervisor/conf.d/mbta-exchange.conf << 'S1'
[program:mbta-exchange]
command=/opt/mbta-agentcy/venv/bin/python -m src.exchange_agent.exchange_server
directory=/opt/mbta-agentcy
autostart=true
autorestart=true
stderr_logfile=/var/log/mbta-exchange.err.log
stdout_logfile=/var/log/mbta-exchange.out.log
environment=PYTHONPATH="/opt/mbta-agentcy"
S1

cat > /etc/supervisor/conf.d/mbta-alerts.conf << 'S2'
[program:mbta-alerts]
command=/opt/mbta-agentcy/venv/bin/python -m uvicorn src.agents.alerts.main:app --host 0.0.0.0 --port 8001
directory=/opt/mbta-agentcy
autostart=true
autorestart=true
stderr_logfile=/var/log/mbta-alerts.err.log
stdout_logfile=/var/log/mbta-alerts.out.log
environment=PYTHONPATH="/opt/mbta-agentcy"
S2

cat > /etc/supervisor/conf.d/mbta-planner.conf << 'S3'
[program:mbta-planner]
command=/opt/mbta-agentcy/venv/bin/python -m uvicorn src.agents.planner.main:app --host 0.0.0.0 --port 8002
directory=/opt/mbta-agentcy
autostart=true
autorestart=true
stderr_logfile=/var/log/mbta-planner.err.log
stdout_logfile=/var/log/mbta-planner.out.log
environment=PYTHONPATH="/opt/mbta-agentcy"
S3

cat > /etc/supervisor/conf.d/mbta-stopfinder.conf << 'S4'
[program:mbta-stopfinder]
command=/opt/mbta-agentcy/venv/bin/python -m uvicorn src.agents.stopfinder.main:app --host 0.0.0.0 --port 8003
directory=/opt/mbta-agentcy
autostart=true
autorestart=true
stderr_logfile=/var/log/mbta-stopfinder.err.log
stdout_logfile=/var/log/mbta-stopfinder.out.log
environment=PYTHONPATH="/opt/mbta-agentcy"
S4

cat > /etc/supervisor/conf.d/mbta-frontend.conf << 'S5'
[program:mbta-frontend]
command=/opt/mbta-agentcy/venv/bin/python -m uvicorn src.frontend.chat_server:app --host 0.0.0.0 --port 3000
directory=/opt/mbta-agentcy
autostart=true
autorestart=true
stderr_logfile=/var/log/mbta-frontend.err.log
stdout_logfile=/var/log/mbta-frontend.out.log
environment=PYTHONPATH="/opt/mbta-agentcy"
S5

echo "Starting services..."
supervisorctl reread
supervisorctl update
supervisorctl start all

sleep 20

echo ""
echo "=== Service Status ==="
supervisorctl status

SETUP

# Cleanup
rm "/tmp/$TARBALL_NAME" 2>/dev/null || true

echo ""
echo "🎉 ============================================================================"
echo "🎉 MBTA Agentcy Deployed!"
echo "🎉 ============================================================================"
echo ""
echo "📍 Instance: $INSTANCE_ID | IP: $PUBLIC_IP"
echo "🔑 Password: $ROOT_PASSWORD"
echo ""
echo "🌐 URLs:"
echo "   Exchange:  http://$PUBLIC_IP:8100"
echo "   Frontend:  http://$PUBLIC_IP:3000"
echo "   Jaeger:    http://$PUBLIC_IP:16686"
echo ""
if [ -n "$REGISTRY_URL" ]; then
    echo "📡 Registry: Agents will auto-register with $REGISTRY_URL"
    echo "   Check: curl $REGISTRY_URL/list"
    echo ""
fi
echo "🧪 Test:"
echo "   curl http://$PUBLIC_IP:8100/"
echo ""
echo "📝 Logs:"
echo "   ssh -i $SSH_KEY_LABEL root@$PUBLIC_IP 'tail -f /var/log/mbta-exchange.out.log'"
echo ""
echo "🛑 Delete:"
echo "   linode-cli linodes delete $INSTANCE_ID"
echo ""