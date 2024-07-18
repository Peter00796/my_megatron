#!/bin/bash

set -exuo pipefail

# [NOTES] This script serves as a standard entry point for executing distributed tasks. It is not intended for direct execution. Instead, customize its default settings by crafting a new script that imports this one.
export OMP_NUM_THREADS=8
export CUDA_DEVICE_MAX_CONNECTIONS=1  #PP

MASTER_ADDR=$MASTER_ADDR
MASTER_PORT=$MASTER_PORT
NNODES=$WORLD_SIZE
NODE_RANK=$RANK

FINETUNE_ARGS="--finetune --no-load-rng --no-load-optim"
TARGET_SCRIPT="pretrain_llama.py"

BATCH_SIZE=2
TRAIN_BATCH_SIZE=$((NNODES*BATCH_SIZE*8*4096))

DATA_PATH="/mnt/pengyanxin/Megatron-LLaMA/data/train_text_document"


HIDDEN_SIZE=2560
FFN_HIDDEN_SIZE=7168
NUM_LAYERS=32
NUM_ATTN_HEADS=20
NUM_QUERY_GROUPS=5
VOCAB_SIZE=122879

TRAIN_SAMPLES=153228731  # ~672B

LR=3e-4
LR_DECAY_STYLE="constant"
MIN_LR=3e-5
LR_WARMUP='--lr-warmup-init 3e-5 --lr-warmup-samples 12207030'  # warmup约50B数据

GPUS_PER_NODE=$(nvidia-smi --list-gpus | wc -l)

# Job info
JOB_ID='megrez_3b_dist_627B_from_scratch_relu_gqa'
JOB_FOLDER=/mnt/pengyanxin/megrez_3b
CHECKPOINT_PATH=$JOB_FOLDER/checkpoints
TENSORBOARD_PATH=$JOB_FOLDER/tensorboard
LOG_PATH=$JOB_FOLDER/logs
WANDB_PATH=$JOB_FOLDER/wandb

mkdir -p $CHECKPOINT_PATH $TENSORBOARD_PATH $LOG_PATH $WANDB_PATH
echo "Job folder is set to $JOB_FOLDER, directories are created."


# Parallel config
TP=1 # Tensor parallel
PP=1 # Pipeline parallel

# ROPE_FACTOR=64
ROPE_BASE=5000000

# Compute batch sizes and training iters
SEQ_LENGTH=4095

GLOBAL_BATCH_SIZE=$((TRAIN_BATCH_SIZE/SEQ_LENGTH))
echo "GLOBAL_BATCH_SIZE is set to $GLOBAL_BATCH_SIZE"

TOKENIZER_PATH=/mnt/pengyanxin/megrez_3b/tokenizer  # Null path.
TOKENIZER_TYPE=NullTokenizer

SEED=42

DISTRIBUTED_ARGS="
--nproc_per_node $GPUS_PER_NODE \
--nnodes $NNODES \
--node_rank $NODE_RANK \
--master_addr $MASTER_ADDR \
--master_port $MASTER_PORT"

CLIP_GRAD="1.0"
OPTIMIZER='adam'

GPT_ARGS="
--tensor-model-parallel-size $TP \
--pipeline-model-parallel-size $PP \
--num-layers $NUM_LAYERS \
--hidden-size $HIDDEN_SIZE \
--num-attention-heads $NUM_ATTN_HEADS \
--group-query-attention \
--num-query-groups $NUM_QUERY_GROUPS \
--ffn-hidden-size $FFN_HIDDEN_SIZE \
--seq-length $SEQ_LENGTH \
--max-position-embeddings $SEQ_LENGTH \
--position-embedding-type rope \
--rope-base $ROPE_BASE \
--micro-batch-size $BATCH_SIZE \
--global-batch-size $GLOBAL_BATCH_SIZE \
--make-vocab-size-divisible-by $TP \
--lr $LR \
--lr-decay-style $LR_DECAY_STYLE \
$LR_WARMUP \
--min-lr $MIN_LR \
--normalization RMSNorm \
--norm-epsilon 1e-5 \
--weight-decay 1e-2 \
--relu-with-gate \
--bf16 \
--no-masked-softmax-fusion \
--no-bias-gelu-fusion \
--no-bias-dropout-fusion \
--untie-embeddings-and-output-weights \
--disable-bias-linear \
--clip-grad $CLIP_GRAD \
--reset-position-ids \
--use-flash-attn \
--attention-softmax-in-fp32 \
--transformer-impl local \
--no-check-for-nan-in-loss-and-grad"


# --recompute-granularity full \
# --recompute-method uniform \
# --recompute-num-layers 2 \

DATA_ARGS="
--data-path $DATA_PATH \
--train-samples $TRAIN_SAMPLES \
--tokenizer-type $TOKENIZER_TYPE \
--tokenizer-model $TOKENIZER_PATH \
--tokenizer-eod 1 \
--no-create-attention-mask-in-dataloader \
--vocab-size $VOCAB_SIZE \
--dataloader-type cyclic \
--num-workers 8 \
--split 1000,0,0"

OUTPUT_ARGS="
--log-interval 1 \
--save-interval 5000 \
--eval-interval 1000"

torchrun $DISTRIBUTED_ARGS $TARGET_SCRIPT \
$GPT_ARGS \
$DATA_ARGS \
$OUTPUT_ARGS \
$FINETUNE_ARGS \
--seed $SEED \
--use-distributed-optimizer \
--optimizer $OPTIMIZER \
--distributed-backend nccl \
--save $CHECKPOINT_PATH \
--no-overlap-p2p-communication \
--tensorboard-dir $TENSORBOARD_PATH \
--tensorboard-log-interval 1 \
--tensorboard-queue-size 5 \
--wandb-save-dir $WANDB_PATH \
--log-throughput \
--log-progress \
--log-timers-to-tensorboard \
--log-memory-to-tensorboard 2>&1 | tee $LOG_PATH/log-rank${RANK}.txt