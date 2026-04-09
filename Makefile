TPU_NAME=node-1
ZONE=us-central2-b
PROJECT=tpu-builder1

# Create a v6e-1 TPU VM (for pivot/smoke test)
create:
	gcloud compute tpus tpu-vm create $(TPU_NAME) \
	    --accelerator-type=v6e-1 --version=v2-alpha \
	    --zone=$(ZONE) --project=$(PROJECT)

# Copy all training scripts and data to the TPU
push:
	gcloud compute tpus tpu-vm scp --recurse \
	    training/ $(TPU_NAME):~/scot/training/ --zone=$(ZONE) --project=$(PROJECT)
	gcloud compute tpus tpu-vm scp --recurse \
	    data/full_run/ $(TPU_NAME):~/scot/data/full_run/ --zone=$(ZONE) --project=$(PROJECT)
	gcloud compute tpus tpu-vm scp \
	    .env $(TPU_NAME):~/scot/.env --zone=$(ZONE) --project=$(PROJECT)
	gcloud compute tpus tpu-vm scp --recurse \
	    data/gsm/ $(TPU_NAME):~/scot/data/gsm/ --zone=$(ZONE) --project=$(PROJECT)

# SSH into the TPU interactively
ssh:
	gcloud compute tpus tpu-vm ssh $(TPU_NAME) --zone=$(ZONE) --project=$(PROJECT)

# Delete the TPU when done
delete:
	gcloud compute tpus tpu-vm delete $(TPU_NAME) \
	    --zone=$(ZONE) --project=$(PROJECT) --quiet
