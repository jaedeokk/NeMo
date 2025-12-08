#!/bin/bash

# Configuration variables
NUM_PROC_PER_NODE=1  # Set the number of processes per node

# Parallelism configruration
DEVICES=1
TP_SIZE=1 ### TP_1 TODO + Virtual PP
PP_SIZE=1 ### PP_4 
# CP_SIZE=2


# Exp logging path
CKPT_DIR="./experiments_finetune/baseline/Qwen2VL_baseline_finetune_2B1_CP1_MBS2_GBS8_seqpack_/Qwen2VL_baseline_finetune_2B1_CP1_MBS2_GBS8_seqpack_--reduced_train_loss=0.6073-epoch=1-consumed_samples=1016000.0"
# CKPT_DIR="PATH_TO_YOUR_WEIGHT"

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
