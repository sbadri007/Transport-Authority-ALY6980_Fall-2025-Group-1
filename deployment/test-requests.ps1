# Test MBTA Agntcy System (Windows)
# Owner: mani

$baseUrl = "http://localhost:8100"

Write-Host "🧪 Testing MBTA Agntcy System" -ForegroundColor Cyan
Write-Host "Owner: mani" -ForegroundColor Cyan
Write-Host ""

# Test 1: General greeting
Write-Host "Test 1: General greeting" -ForegroundColor Yellow
$body1 = @{
    message = "Hello! How are you?"
    user_id = "test_user"
} | ConvertTo-Json

try {
    $response1 = Invoke-RestMethod -Uri "$baseUrl/chat" -Method Post -Body $body1 -ContentType "application/json"
    $response1 | ConvertTo-Json -Depth 10
} catch {
    Write-Host "❌ Test 1 failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "---" -ForegroundColor Gray
Write-Host ""

# Test 2: MBTA alerts query
Write-Host "Test 2: MBTA alerts query" -ForegroundColor Yellow
$body2 = @{
    message = "Are there any delays on the Red Line?"
    user_id = "test_user"
} | ConvertTo-Json

try {
    $response2 = Invoke-RestMethod -Uri "$baseUrl/chat" -Method Post -Body $body2 -ContentType "application/json"
    $response2 | ConvertTo-Json -Depth 10
} catch {
    Write-Host "❌ Test 2 failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "---" -ForegroundColor Gray
Write-Host ""

# Test 3: Trip planning
Write-Host "Test 3: Trip planning query" -ForegroundColor Yellow
$body3 = @{
    message = "How do I get from Harvard to MIT?"
    user_id = "test_user"
} | ConvertTo-Json

try {
    $response3 = Invoke-RestMethod -Uri "$baseUrl/chat" -Method Post -Body $body3 -ContentType "application/json"
    $response3 | ConvertTo-Json -Depth 10
} catch {
    Write-Host "❌ Test 3 failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "---" -ForegroundColor Gray
Write-Host ""

# Test 4: Stop search
Write-Host "Test 4: Stop search query" -ForegroundColor Yellow
$body4 = @{
    message = "Find stops near Kendall Square"
    user_id = "test_user"
} | ConvertTo-Json

try {
    $response4 = Invoke-RestMethod -Uri "$baseUrl/chat" -Method Post -Body $body4 -ContentType "application/json"
    $response4 | ConvertTo-Json -Depth 10
} catch {
    Write-Host "❌ Test 4 failed: $_" -ForegroundColor Red
}

Write-Host ""
Write-Host "✅ Tests complete!" -ForegroundColor Green