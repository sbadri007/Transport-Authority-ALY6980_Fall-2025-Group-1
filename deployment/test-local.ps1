# MBTA Agntcy - Local Testing Script (Windows)
# Owner: mani

Write-Host "🧪 MBTA Agntcy - Local Testing" -ForegroundColor Cyan
Write-Host "Owner: mani" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "🐍 Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version
    Write-Host "✅ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check ANTHROPIC_API_KEY
Write-Host "🔑 Checking environment variables..." -ForegroundColor Yellow
if (-not $env:ANTHROPIC_API_KEY) {
    Write-Host "⚠️  ANTHROPIC_API_KEY not set" -ForegroundColor Yellow
    Write-Host "Please run:" -ForegroundColor Yellow
    Write-Host '  $env:ANTHROPIC_API_KEY = "your-key-here"' -ForegroundColor Cyan
    exit 1
}
Write-Host "✅ API key is set" -ForegroundColor Green

# Create virtual environment
Write-Host "📦 Setting up virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path "venv")) {
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Install dependencies
Write-Host "📚 Installing dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Create logs directory
if (-not (Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" | Out-Null
}

Write-Host ""
Write-Host "🚀 Starting services..." -ForegroundColor Cyan
Write-Host ""

# Start Exchange Agent
Write-Host "▶️  Starting Exchange Agent (port 8100)..." -ForegroundColor Green
$exchangeJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & .\venv\Scripts\python.exe -m uvicorn src.exchange_agent.exchange_server:app --host 0.0.0.0 --port 8100
}
Write-Host "   Job ID: $($exchangeJob.Id)" -ForegroundColor Gray
Start-Sleep -Seconds 3

# Start MBTA Orchestrator
Write-Host "▶️  Starting MBTA Orchestrator (port 8101)..." -ForegroundColor Green
$orchestratorJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & .\venv\Scripts\python.exe -m uvicorn src.orchestrator.mbta_server:app --host 0.0.0.0 --port 8101
}
Write-Host "   Job ID: $($orchestratorJob.Id)" -ForegroundColor Gray
Start-Sleep -Seconds 3

# Start Frontend
Write-Host "▶️  Starting Frontend (port 3000)..." -ForegroundColor Green
$frontendJob = Start-Job -ScriptBlock {
    Set-Location $using:PWD
    & .\venv\Scripts\python.exe -m uvicorn src.frontend.chat_server:app --host 0.0.0.0 --port 3000
}
Write-Host "   Job ID: $($frontendJob.Id)" -ForegroundColor Gray
Start-Sleep -Seconds 5

# Health checks
Write-Host ""
Write-Host "🏥 Running health checks..." -ForegroundColor Yellow

function Test-ServiceHealth {
    param($name, $url)
    try {
        $response = Invoke-WebRequest -Uri $url -Method Get -TimeoutSec 5 -ErrorAction Stop
        Write-Host "✅ $name is healthy" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ $name health check failed" -ForegroundColor Red
        return $false
    }
}

Test-ServiceHealth "Exchange Agent" "http://localhost:8100/health"
Test-ServiceHealth "MBTA Orchestrator" "http://localhost:8101/health"
Test-ServiceHealth "Frontend" "http://localhost:3000/health"

# Save job IDs
"$($exchangeJob.Id),$($orchestratorJob.Id),$($frontendJob.Id)" | Out-File -FilePath ".job_ids"

Write-Host ""
Write-Host "🎉 All services started!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Service URLs:" -ForegroundColor Cyan
Write-Host "   - Frontend UI:        http://localhost:3000" -ForegroundColor White
Write-Host "   - Exchange Agent:     http://localhost:8100" -ForegroundColor White
Write-Host "   - MBTA Orchestrator:  http://localhost:8101" -ForegroundColor White
Write-Host ""
Write-Host "📊 API Documentation:" -ForegroundColor Cyan
Write-Host "   - Exchange Agent:     http://localhost:8100/docs" -ForegroundColor White
Write-Host "   - MBTA Orchestrator:  http://localhost:8101/docs" -ForegroundColor White
Write-Host ""
Write-Host "📝 View Logs:" -ForegroundColor Cyan
Write-Host "   Get-Job | Receive-Job" -ForegroundColor White
Write-Host ""
Write-Host "🛑 To stop all services:" -ForegroundColor Yellow
Write-Host "   .\deployment\stop-local.ps1" -ForegroundColor White
Write-Host ""
Write-Host "🧪 Run tests:" -ForegroundColor Cyan
Write-Host "   .\test_requests.ps1" -ForegroundColor White