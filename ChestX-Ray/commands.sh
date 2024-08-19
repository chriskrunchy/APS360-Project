# pretrained resnet
python pretrained_resnet18.py --batch_size 32 --num_workers 4 --gpus "0,1" --num_epochs 100
python pretrained_resnet34.py --batch_size 32 --num_workers 4 --gpus "2,3" --num_epochs 100
python pretrained_resnet50.py --batch_size 32 --num_workers 4 --gpus "0,1" --num_epochs 100
python pretrained_resnet152.py --batch_size 32 --num_workers 4 --gpus "0,1" --num_epochs 100


# efficientNet (pretrained)
python pretrained_efficientnet.py --batch_size 32 --num_workers 4 --gpus "0,1" --num_epochs 100

# cheXNet (pretrained DenseNet121)
python pretrained_chexnet.py --batch_size 32 --num_workers 4 --gpus "2,3" --num_epochs 100

# ViT-Base 
python pretrained_vit_base.py --batch_size 32 --num_workers 4 --gpus "2,3" --num_epochs 100