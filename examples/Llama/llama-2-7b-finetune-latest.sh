#!/bin/bash 

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=4
MASTER_ADDR=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

CHECKPOINT_PATH=/mnt/pengyanxin/checkpoints
DATA_PATH=/mnt/pengyanxin/data/output/output_text_document
TOKENIZER_MODEL=/mnt/pengyanxin/megatron-ckpts
CHECKPOINT_DIR=/mnt/pengyanxin/checkpoints
TP=1
PP=4

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
LLAMA_ARGS="
    --num-layers 48 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --micro-batch-size 2 \
    --global-batch-size 8 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --lr 1.5e-5 \
    --train-iters 500000 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-6 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --load ${CHECKPOINT_DIR} \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
    --no-load-optim \
    --no-load-rng \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --normalization RMSNorm \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --attention-softmax-in-fp32
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"

OUTPUT_ARGS="
    --log-interval 1 \
    --save-interval 500 \
    --eval-interval 100 \
    --eval-iters 100
"

torchrun $DISTRIBUTED_ARGS /mnt/pengyanxin/my_megatron/pretrain_gpt.py \
    $LLAMA_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH