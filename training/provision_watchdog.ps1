param (
    [string]$Zone = "us-east5-a",
    [string]$Project = "tpu-builder1",
    [string]$TpuName = "my-tpu-node",
    [int]$IntervalSeconds = 120
)

Write-Host "🚀 Starting TPU Provisioning Watchdog for $TpuName in $Zone..." -ForegroundColor Cyan

while ($true) {
    Write-Host "$(Get-Date -Format 'HH:mm:ss'): Attempting to provision v5p-8..." -NoNewline
    
    # Try to delete any failed/existing queue entry first
    gcloud alpha compute tpus queued-resources delete tpu-builder-queue --zone=$Zone --project=$Project --force --quiet 2>$null | Out-Null
    
    # Attempt to create the Queued Resource
    $result = gcloud alpha compute tpus queued-resources create tpu-builder-queue `
        --accelerator-type=v5p-8 --runtime-version=v2-alpha-tpuv5 `
        --zone=$Zone --project=$Project --node-id=$TpuName `
        --valid-until-duration=14400s --quiet 2>&1
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host " ✅ REQUEST ACCEPTED!" -ForegroundColor Green
        Write-Host "TPU is now in the queue. Monitoring state..."
        
        while ($true) {
            $state = gcloud alpha compute tpus queued-resources describe tpu-builder-queue --zone=$Zone --project=$Project --format="value(state.state)" 2>$null
            Write-Host "$(Get-Date -Format 'HH:mm:ss'): Current State: $state"
            
            if ($state -eq "PROVISIONING" -or $state -eq "ACTIVE" -or $state -eq "ACCEPTED" -or $state -eq "WAITING_FOR_RESOURCES") {
                if ($state -eq "ACTIVE") {
                    Write-Host "✨ TPU IS ACTIVE! Ready for training." -ForegroundColor Green
                    exit 0
                }
            } else {
                Write-Host "❌ Request state shifted to $state. Retrying main loop..." -ForegroundColor Yellow
                break
            }
            Start-Sleep -Seconds 60
        }
    } else {
        Write-Host " ❌ Exhausted (Status $LASTEXITCODE). Retrying in $IntervalSeconds seconds..." -ForegroundColor Gray
    }
    
    Start-Sleep -Seconds $IntervalSeconds
}
