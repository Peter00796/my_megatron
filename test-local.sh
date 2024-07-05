rm -rf /mnt/pengyanxin/mdl/mizar-checkpoint-engine/dlrover/python/elastic_agent/torch/failure_flag

cd /mnt/pengyanxin/mdl/my_megatron/examples/gpt3
echo 50 > ../../checkpoints/dlrover_latest.txt
echo 50 > ../../checkpoints/latest_checkpointed_iteration.txt
sh train_gpt_345m_distributed.sh | tee -a /mnt/pengyanxin/mdl/logs/train_gpt_345m_distributed.log
