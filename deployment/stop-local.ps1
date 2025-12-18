# Stop local services (Windows)

Write-Host "🛑 Stopping MBTA Agntcy services..." -ForegroundColor Yellow

if (Test-Path ".job_ids") {
    $jobIds = (Get-Content ".job_ids").Split(",")
    
    foreach ($jobId in $jobIds) {
        Write-Host "Stopping Job ID: $jobId..." -ForegroundColor Gray
        Stop-Job -Id $jobId -ErrorAction SilentlyContinue
        Remove-Job -Id $jobId -ErrorAction SilentlyContinue
    }
    
    Remove-Item ".job_ids"
    Write-Host "✅ All services stopped" -ForegroundColor Green
} else {
    Write-Host "⚠️  No job IDs file found. Services may not be running." -ForegroundColor Yellow
}