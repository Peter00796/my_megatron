rm -rf /mnt/pengyanxin/my_megatron/examples/gpt3/gpt2_345m/iter_*
rm -rf /mnt/pengyanxin/my_megatron/examples/gpt3/gpt2_345m/latest_info.txt
rm -rf /mnt/pengyanxin/my_megatron/examples/gpt3/gpt2_345m/latest_checkpointed_iteration.txt
touch /mnt/pengyanxin/my_megatron/examples/gpt3/gpt2_345m/latest_checkpointed_iteration.txt
echo "release" >> /mnt/pengyanxin/my_megatron/examples/gpt3/gpt2_345m/latest_checkpointed_iteration.txt
sh /mnt/pengyanxin/my_megatron/examples/gpt3/KP_exp_1GPU_345M_GPT.sh