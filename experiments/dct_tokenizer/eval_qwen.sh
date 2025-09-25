#!/bin/bash

# Configuration variables
NUM_PROC_PER_NODE=4  # Set the number of processes per node

# Parallelism configruration
DEVICES=4
TP_SIZE=2 ### TP_1 TODO + Virtual PP
PP_SIZE=2 ### PP_4 
# CP_SIZE=2


# Exp logging path
# CKPT_DIR="/workspace/experiments_finetune/baseline/Qwen2VL_baseline_finetune_2B2_CP_MBS1_GBS4_seqpack_/Qwen2VL_baseline_finetune_2B2_CP_MBS1_GBS4_seqpack_--reduced_train_loss=3.5755-epoch=1-consumed_samples=20760.0-last/weights"
CKPT_DIR="PATH_TO_YOUR_WEIGHT"

# Construct the arguments stringring
ARGS=(
  # "--load_from_hf" "$DATA_TYPE"
  "--local_model_path" "$CKPT_DIR"
  "--tp_size" "$TP_SIZE"
  "--pp_size" "$PP_SIZE"
)

# Run the experiment with torchrun
echo "NUM_PROC_PER_NODE: ${NUM_PROC_PER_NODE}"
echo "ARGS: ${ARGS[@]}"

torchrun --nproc_per_node=$NUM_PROC_PER_NODE \
     --master_port=29501 \
    /workspace/scripts/vlm/qwen2vl_generate.py ${ARGS[@]}
