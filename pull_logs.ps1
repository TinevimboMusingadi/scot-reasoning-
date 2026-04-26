# pull_logs.ps1
# Run this locally in PowerShell to periodically pull the logs from the TPU
$TpuName = "my-tpu-node"
$Zone = "us-east5-a"
$Project = "tpu-builder1"

Write-Host "Started Log Watcher. Press Ctrl+C to stop."
Write-Host "Polling every 30 seconds..."

# Ensure local logs directory exists
if (-Not (Test-Path -Path .\logs)) {
    New-Item -ItemType Directory -Path .\logs | Out-Null
}

while ($true) {
    # Pull S-CoT log if it exists
    gcloud compute tpus tpu-vm scp --project=$Project --zone=$Zone ${TpuName}:~/scot/scot_train.log .\logs\scot_train.log --quiet 2>$null
    
    # Pull Flat log if it exists
    gcloud compute tpus tpu-vm scp --project=$Project --zone=$Zone ${TpuName}:~/scot/flat_train.log .\logs\flat_train.log --quiet 2>$null
    
    Write-Host "Synced logs at $(Get-Date -Format 'HH:mm:ss')"
    
    Start-Sleep -Seconds 30
}
