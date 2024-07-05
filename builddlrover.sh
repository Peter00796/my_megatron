
cd /mnt/pengyanxin/mdl/mizar-checkpoint-engine
pip uninstall dlrover
python setup.py sdist bdist_wheel
pip install dist/dlrover-0.3.7-py3-none-any.whl
echo "install dlrover successfully"