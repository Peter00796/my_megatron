cd /mnt/pengyanxin/mdl/mizar-checkpoint-engine

rm -f dlrover/python/elastic_agent/torch/failure_flag

cd /mnt/pengyanxin/mdl/my_megatron/examples/gpt3
sh train_gpt_345m_distributed.sh | tee -a /mnt/pengyanxin/mdl/logs/train_gpt_345m_distributed.log
