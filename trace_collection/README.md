Here we detail the steps to collect traces for the CS5470 project. We make the complicated process of spinning up Google Cloud TPUs, attaching and mounting external storage (needed for large model sizes), and running the trace collection script as simple as possible.

## Step 1: Spin up a Google Cloud TPU
Here's an example command to spin up a TPU v6e-8:
`gcloud alpha compute tpus queued-resources create my-tpu-v6e-queue --node-id my-tpu-v6e --zone=us-east1-d --accelerator-type=v6e-8 --runtime-version=v2-alpha-tpuv6e --network=global-project-net  --subnetwork=global-project-net --provisioning-model=SPOT`

## Step 2: Allocate external storage
`gcloud compute disks create hf-disk --type=hyperdisk-balanced --size=300GB --zone=us-east1-d`

## Step 3: Run our TPU setup script. It handles mounting the external storage and sets up Docker for running profiling. 

Usage: `./profiling_setup.sh <ZONE> <TPU_NAME> <DISK_NAME>` e.g. `./profiling_setup.sh us-east1-d  my-tpu-v6e  hf-disk`

## Step 4: Run our trace collection script
We wrote a trace collection script that collects all 88 traces in one go while being robust to errors and potential preemption of TPUs. When rerun, it only performs profiling for traces that haven't been captured previously. Traces are automatically SCPed from the Docker container to the host TPU and then to your local machine and are placed in a folder that details the trace configuration in its name (e.g MODEL_meta-llama_Llama-3.1-8B,INPUT_1024,OUTPUT_1,BATCH_4,TP_1).

Usage: `python run_all_traces.py <ZONE> <TPU_NAME>`
eg. `python run_all_traces.py us-east1-d  my-tpu-v6e`                

Running a single script can be done quickly by just setting 
        `models,` `tp_sizes` and `batch_sizes` to arrays with one element, or you can directly SSH into the TPU and run 

    export DOCKER_URI=vllm/vllm-tpu:v0.12.0
        export HF_HOME=/mnt/disks/huggingface/
        sudo docker run -it --rm --name $USER-vllm --privileged --net=host \
        -v /mnt/disks/huggingface:/mnt/disks/huggingface\
    -v /dev/shm:/dev/shm \
    --shm-size 150gb \
    --entrypoint /bin/bash ${DOCKER_URI}
