# pretrained-resnet
python pretrained_resnet.py --batch_size 32 --num_workers 4 --gpus "0,1" --num_epochs 100

# pretrained-vgg
python pretrained_vgg.py --batch_size 32 --num_workers 4 --gpus "0,1" --num_epochs 100

# pretrained-cheXNet
python pretrained_chexnet.py --batch_size 32 --num_workers 4 --gpus "2,3" --num_epochs 100

# vit-base
python pretrained_vit_base.py --batch_size 32 --num_workers 4 --gpus "2,3" --num_epochs 100
