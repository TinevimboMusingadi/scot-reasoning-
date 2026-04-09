param (
    [string]$Zone = "us-east5-a",
    [string]$Project = "tpu-builder1",
    [string]$TpuName = "my-tpu-node",
    [int]$IntervalSeconds = 120
)

Write-Host "[Watchdog] Starting Flex-Start Watchdog for $TpuName..." -ForegroundColor Cyan

while ($true) {
    # 1. Check current state
    $desc = gcloud alpha compute tpus queued-resources describe tpu-builder-queue --zone=$Zone --project=$Project --format="value(state.state)" 2>$null
    
    if ($null -eq $desc -or $desc -eq "") {
        Write-Host "$(Get-Date -Format 'HH:mm:ss'): No active resource found. Re-submitting create request..." -ForegroundColor Yellow
        
        # Submitting the EXACT command that worked for you
        gcloud alpha compute tpus queued-resources create tpu-builder-queue `
            --zone=$Zone `
            --accelerator-type=v5p-8 `
            --runtime-version=v2-alpha-tpuv5 `
            --node-id=$TpuName `
            --provisioning-model=flex-start `
            --max-run-duration=4h `
            --valid-until-duration=4h `
            --labels=purpose=flex-start --quiet 2>$null | Out-Null
            
        if ($LASTEXITCODE -eq 0) {
            Write-Host " [Success] Create Request Re-issued." -ForegroundColor Green
        } else {
            Write-Host " [Pending] Cloud still busy. Retrying in $IntervalSeconds seconds..." -ForegroundColor Gray
        }
    } 
    elseif ($desc -eq "SUSPENDING" -or $desc -eq "SUSPENDED") {
        Write-Host "$(Get-Date -Format 'HH:mm:ss'): Current State: $desc. Waiting for system to clear slot..." -ForegroundColor Yellow
        # If suspended, we need to delete it so we can re-create it
        if ($desc -eq "SUSPENDED") {
            Write-Host "[Cleanup] Cleaning up suspended resource..."
            gcloud alpha compute tpus queued-resources delete tpu-builder-queue --zone=$Zone --project=$Project --force --quiet 2>$null | Out-Null
        }
    }
    elseif ($desc -eq "PROVISIONING" -or $desc -eq "ACTIVE" -or $desc -eq "ACCEPTED" -or $desc -eq "WAITING_FOR_RESOURCES") {
        Write-Host "$(Get-Date -Format 'HH:mm:ss'): Current State: $desc. Monitoring..." -ForegroundColor Green
        if ($desc -eq "ACTIVE") {
            Write-Host "[Success] TPU IS LIVE! Ready for training." -ForegroundColor Green
            exit 0
        }
    }
    else {
        Write-Host "$(Get-Date -Format 'HH:mm:ss'): Resource state is $desc."
    }

    Start-Sleep -Seconds $IntervalSeconds
}
