# Baseline
torchrun --nnodes=1 --nproc_per_node=2 baseline.py 
python baseline.py --gpus 0,1