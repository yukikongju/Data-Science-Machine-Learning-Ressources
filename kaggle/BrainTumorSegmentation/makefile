PROJECT_ID=relax-server
TPU_NAME=node1
# ZONE=us-east5-a
# ZONE=us-east1-b 
ZONE=us-central1-a # unavailable for TPU
# ZONE=us-east5-a # single tenant
# ZONE=us-west1-c
# TPU_ACCELERATOR_TYPE=v5litepod-8
TPU_ACCELERATOR_TYPE=v5litepod-256
RUNTIME_VERSION=v2-alpha-tpuv5-lite

RESERVATION=test-reservation
GPU_MACHINE_TYPE=n1-standard-4
MACHINE_TYPE=ct5e-hightpu-1t # version: 5; core:4
RESERVATION_ACCELERATOR_TYPE=nvidia-tesla-t4
ACCELERATOR_COUNT=8
VM_COUNT=1
VM_FAMILY=tpus

GPU_NAME=ds-gpu
GPU_TYPE=nvidia-tesla-t4
GPU_COUNT=1


list-tpus:
	gcloud compute tpus tpu-vm list --zone=$(ZONE)

remove-tpu:
	echo

create-tpu:
	gcloud compute tpus tpu-vm create $(TPU_NAME) \
	    --project=$(PROJECT_ID) \
	    --zone=$(ZONE) \
	    --accelerator-type=$(TPU_ACCELERATOR_TYPE) \
	    --version=$(RUNTIME_VERSION)

tpu-connect:
	gcloud compute tpus tpu-vm ssh $(TPU_NAME) \
	    --project=$(PROJECT_ID) \
	    --zone=$(ZONE)

create-reservation:
	gcloud alpha compute reservations create $(RESERVATION) \
		--zone=$(ZONE) \
		--machine-type=$(MACHINE_TYPE) \
		--vm-count=1 \
		# --accelerator-count=$(ACCELERATOR_COUNT) \
		# --vm-family=$(VM_FAMILY)
		# --accelerator-type=$(ACCELERATOR_TYPE) \

list-region:
	gcloud compute regions list \
	  --project=$(PROJECT_ID) \
	  --format="csv(name, quotas.metric, quotas.limit, quotas.usage, quotas.region)" \
	  --filter="quotas.limit > 0"

	# gcloud compute regions list --project=relax-server --format="table(name, quotas.metric, quotas.limit, quotas.usage)" --filter="quotas.metric~TPU"

	# gcloud compute accelerator-types list --filter="name~'nvidia'" --sort-by=zone
create-gpu:
	gcloud compute instances create $(GPU_NAME) \
	  --project=$(PROJECT_ID) \
	  --zone=$(ZONE) \
	  --machine-type=$(GPU_MACHINE_TYPE) \
	  --accelerator type=$(GPU_TYPE),count=$(GPU_COUNT) \
	  --maintenance-policy=TERMINATE \
	  --restart-on-failure \
	  --preemptible 
	  # --image-family=common-cu110 \
	  # --image-project=deeplearning-platform-release
	# auto shut-down after 3 min idle
	gcloud compute instances add-metadata $(GPU_NAME) \
	  --metadata=startup-script='
	#!/bin/bash
	IDLE_LIMIT=180
	while true; do
	  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
	  if [[ $GPU_UTIL -eq 0 ]]; then
	    ((IDLE+=60))
	    if [[ $IDLE -ge $IDLE_LIMIT ]]; then
	      sudo shutdown -h now
	    fi
	  else
	    IDLE=0
	  fi
	  sleep 60
	done
	'

list-gpu:
	gcloud compute instances list

stop-gpu:
	gcloud compute instances stop $(GPU_NAME) --zone=$(ZONE)

delete-gpu:
	gcloud compute instances delete $(GPU_NAME) --zone=$(ZONE)


