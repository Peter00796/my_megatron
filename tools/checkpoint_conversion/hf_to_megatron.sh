python tools/checkpoint_conversion/llama_checkpoint_conversion.py \
--load_path "/mnt/resource/public_models/Llama-2-7b-hf" \
--save_path "/mnt/public/pengyanxin/workspace/my_megatron/megatron-ckpts" \
--target_tensor_model_parallel_size 1 \
--target_pipeline_model_parallel_size 4 \
--target_data_parallel_size 1 \
--target_params_dtype "fp16" \
--make_vocab_size_divisible_by 1 \
--print-checkpoint-structure \
--megatron-path "/mnt/public/pengyanxin/workspace/my_megatron"