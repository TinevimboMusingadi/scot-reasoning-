import subprocess
import time
import sys
import json
import os

# --- CONFIGURATION ---
# Try different zones if one is stuck. Available options from TPU Builders:
#   v5p: us-east5-a (accelerator-type=v5p-8, runtime=v2-alpha-tpuv5)
#   v6e: us-central1-a (accelerator-type=v6e-8, runtime=v2-alpha-tpuv6e)
#   v6e: southamerica-east1-c (accelerator-type=v6e-8, runtime=v2-alpha-tpuv6e)
#   v5e: us-west4-a (accelerator-type=v5litepod-8, runtime=v2-alpha-tpuv5-lite)

ZONE = "us-central1-a"
PROJECT = "tpu-builder1"
QUEUE_NAME = "tpu-builder-queue"
NODE_NAME = "my-tpu-node"
ACCEL_TYPE = "v6e-8"
RUNTIME = "v2-alpha-tpuv6e"
GCLOUD_CMD = "gcloud.cmd" if os.name == 'nt' else "gcloud"

def describe_queue():
    cmd = [GCLOUD_CMD, "alpha", "compute", "tpus", "queued-resources", "describe", QUEUE_NAME, f"--zone={ZONE}", f"--project={PROJECT}", "--format=json"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return data.get('state', {}).get('state', 'UNKNOWN')
    except subprocess.CalledProcessError as e:
        if "not found" in e.stderr.lower() or "not_found" in e.stderr.lower():
            return "NOT_FOUND"
        return "ERROR: " + e.stderr

def delete_queue():
    print(f"[*] Deleting suspended/failed queue {QUEUE_NAME}...")
    cmd = [GCLOUD_CMD, "alpha", "compute", "tpus", "queued-resources", "delete", QUEUE_NAME, f"--zone={ZONE}", f"--project={PROJECT}", "--force", "--quiet"]
    subprocess.run(cmd, capture_output=True, text=True)

def delete_old_queue():
    """Delete the old queue in the previous zone if it exists."""
    old_zone = "us-east5-a"
    print(f"[*] Cleaning up old queue in {old_zone}...")
    cmd = [GCLOUD_CMD, "alpha", "compute", "tpus", "queued-resources", "delete", QUEUE_NAME, f"--zone={old_zone}", f"--project={PROJECT}", "--force", "--quiet"]
    subprocess.run(cmd, capture_output=True, text=True)

def create_queue():
    print(f"[*] Creating flex-start queue {QUEUE_NAME} in {ZONE} ({ACCEL_TYPE}, max 4h)...")
    cmd = [
        GCLOUD_CMD, "alpha", "compute", "tpus", "queued-resources", "create", QUEUE_NAME,
        f"--node-id={NODE_NAME}", f"--zone={ZONE}", f"--project={PROJECT}",
        f"--accelerator-type={ACCEL_TYPE}", f"--runtime-version={RUNTIME}",
        "--provisioning-model=flex-start", "--max-run-duration=4h"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[!] Queue creation output: {result.stderr}")

def launch_training():
    print(f"[*] TPU is ACTIVE! Waiting 45s for boot sequences...")
    time.sleep(45)
    
    print("[*] Streaming Training Execute Command over SSH...")
    ssh_cmd = [
        GCLOUD_CMD, "compute", "tpus", "tpu-vm", "ssh", NODE_NAME,
        f"--zone={ZONE}", f"--project={PROJECT}",
        "--command", "if [ ! -d ~/scot ]; then git clone https://github.com/TinevimboMusingadi/scot-reasoning-.git ~/scot; else cd ~/scot && git pull; fi && cd ~/scot && bash training/setup_tpu.sh && bash training/run_all.sh"
    ]
    process = subprocess.Popen(ssh_cmd, stdout=sys.stdout, stderr=sys.stderr)
    process.wait()
    print(f"[*] SSH session dropped with code {process.returncode}")
    return process.returncode

def main():
    print("=========================================")
    print("      SFT Distillation Watchdog          ")
    print(f"  Zone: {ZONE} | TPU: {ACCEL_TYPE}")
    print("=========================================")
    
    # Clean up old queue in previous zone first
    delete_old_queue()
    
    while True:
        state = describe_queue()
        
        if state == "ACTIVE":
            exit_code = launch_training()
            if exit_code == 0:
                print("[!] Watchdog completed cleanly. Run finished successfully!")
                break
            else:
                print(f"[*] Run disconnected (Preemption/Error). Re-verifying node state...")
                time.sleep(15)
        elif state == "NOT_FOUND":
            create_queue()
            time.sleep(20)
        elif state in ["SUSPENDED", "FAILED"]:
            delete_queue()
            time.sleep(30)
        elif state.startswith("ERROR"):
            print(f"Error querying state: {state}")
            time.sleep(30)
        else:
            print(f"[*] Status: {state}. Waiting in queue...")
            time.sleep(45)

if __name__ == '__main__':
    main()
