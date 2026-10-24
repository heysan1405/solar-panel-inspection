$ErrorActionPreference = 'Stop'

$port = 8000
$pythonExe = 'C:/Users/PEXUS/AppData/Local/Programs/Python/Python313/python.exe'
$listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1

if ($listener) {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:$port/api/health" -TimeoutSec 10
        Write-Host "PVEL-AD backend is already running on http://127.0.0.1:$port" -ForegroundColor Green
        Write-Host "Detector ready: $($health.detector); Ollama ready: $($health.ollama)"
        exit 0
    } catch {
        throw "Port $port is already used by PID $($listener.OwningProcess), but it is not responding as the PVEL-AD API. Stop that process manually or choose another port."
    }
}

Write-Host "Starting PVEL-AD backend on http://127.0.0.1:$port" -ForegroundColor Cyan
& $pythonExe -m uvicorn backend:app --host 127.0.0.1 --port $port
