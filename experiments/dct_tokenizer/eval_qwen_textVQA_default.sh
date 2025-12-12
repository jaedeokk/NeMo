#!/bin/bash

# Configuration variables
NUM_PROC_PER_NODE=1  # Set the number of processes per node
# CP_SIZE=2
TP_SIZE=1 ### 
PP_SIZE=1 ### PP_4 

IMAGE_PATH='/datasets/TextVQA/val/train_images/'
JSON_PATH='/datasets/TextVQA/val/TextVQA_0.5.1_val.json'

MODEL_SIZE="2B"
OUTPUT_PATH="/datasets/TextVQA/results/pred_textvqa_val_${MODEL_SIZE}.json"
OSL=10
# Exp logging path
# Construct the arguments stringring
ARGS=(
  "--load_from_hf"
  "--tp_size" "$TP_SIZE"
  "--pp_size" "$PP_SIZE"
  "--image_folder" "$IMAGE_PATH"
  "--json_path" "$JSON_PATH"
  "--output_json" "$OUTPUT_PATH"
  "--model_size" "$MODEL_SIZE"
  "--osl" $OSL
  )

# Run the experiment with torchrun

echo "NUM_PROC_PER_NODE: ${NUM_PROC_PER_NODE}"
echo "ARGS: ${ARGS[@]}"

torchrun --nproc_per_node=$NUM_PROC_PER_NODE \
    /workspace/scripts/vlm/qwen2vl_textVQA_eval.py ${ARGS[@]}
