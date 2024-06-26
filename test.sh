cd /mnt/pengyanxin/mdl/mizar-chekpoint-engine

pip uninstall dlrover
Y
python setup.py sdist bdist_wheel
pip install dist/dlrover-0.3.7-py3-none-any.whl
echo "install dlrover successfully"

cd /mnt/pengyanxin/mdl/my_megatron
pip install -e .
export PATH=$PATH:~/.local/bin

cd /mnt/pengyanxin/mdl/my_megatron/examples/gpt3
sh train_gpt_345m_distributed.sh | tee -a /mnt/pengyanxin/mdl/logs/train_gpt_345m_distributed.log