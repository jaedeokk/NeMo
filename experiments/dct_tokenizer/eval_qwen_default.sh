#!/bin/bash

# Configuration variables
NUM_PROC_PER_NODE=1  # Set the number of processes per node
# CP_SIZE=2
TP_SIZE=1 ### 
PP_SIZE=1 ### PP_4 
IMAGE_PATH='/datasets/TextVQA/val/train_images/5ce862cbefd8458f.jpg'

# Exp logging path
# Construct the arguments stringring
ARGS=(
  "--load_from_hf"
  "--image_url" "$IMAGE_PATH"
  "--tp_size" "$TP_SIZE"
  "--pp_size" "$PP_SIZE"
    )

# Run the experiment with torchrun

echo "NUM_PROC_PER_NODE: ${NUM_PROC_PER_NODE}"
echo "ARGS: ${ARGS[@]}"

torchrun --nproc_per_node=$NUM_PROC_PER_NODE \
    /workspace/scripts/vlm/qwen2vl_generate.py ${ARGS[@]}
