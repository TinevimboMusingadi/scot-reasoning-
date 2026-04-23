import subprocess
import time
import sys
import json
import os

ZONE = "us-east5-a"
PROJECT = "tpu-builder1"
QUEUE_NAME = "tpu-builder-queue"
NODE_NAME = "my-tpu-node"
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

def create_queue():
    print(f"[*] Creating flex-start queue {QUEUE_NAME} (max 4h)...")
    cmd = [
        GCLOUD_CMD, "alpha", "compute", "tpus", "queued-resources", "create", QUEUE_NAME,
        f"--node-id={NODE_NAME}", f"--zone={ZONE}", f"--project={PROJECT}",
        "--accelerator-type=v5p-8", "--runtime-version=v2-alpha-tpuv5",
        "--provisioning-model=flex-start", "--max-run-duration=4h"
    ]
    subprocess.run(cmd, capture_output=True, text=True)

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
    print("=========================================")
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
