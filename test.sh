cd /mnt/pengyanxin/mizar-checkpoint-engine

python setup.py sdist bdist_wheel
pip install dist/dlrover-0.3.6rc0-py3-none-any.whl
echo "install dlrover successfully"
pip show dlrover

cd /mnt/pengyanxin/Megatron-LM 
pip install -e .
export PATH=$PATH:~/.local/bin

cd /mnt/pengyanxin/Megatron-LM/examples/gpt3
sh train_gpt_345m_distributed.sh









