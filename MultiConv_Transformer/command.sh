# Baseline
torchrun --nnodes=1 --nproc_per_node=2 baseline.py 
python baseline.py --gpus 0,1

# MultiConv_Transformer
python train.py --gpus 0,1,2,3

# resnet
python resnet.py --gpus 0,1,2,3