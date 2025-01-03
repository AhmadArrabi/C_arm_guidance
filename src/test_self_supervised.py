import torch.nn as nn
import torch
from dataset import Positional_dataset
from models import *
import argparse
import sys

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training self-supervised model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='resnet34',
        required=True,
        help=("the name of the backbone used in training")
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default=None,
        required=True,
        help=("Annotations folder")
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="root/logs/self_supervised/experiment_1/checkpoints/chkpt_e100.pt",
        required=True,
        help=(".pth checkpoint for evaluation (full path)")
    )
    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_size = [256,256]

    dataset_val = Positional_dataset(augmentation=False, mode='test', size=img_size, root_annotations=args.annotations_dir)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    
    model = nn.DataParallel(positional_understanding_model(backbone=args.model_name, remove_patient_stats=args.remove_patient_stats))
    model = model.to(DEVICE).to(torch.float32)
        
    checkpoint = torch.load(f'{args.checkpoint_path}')
    state_dict = checkpoint['model_state_dict']
    
    model.load_state_dict(state_dict)
    print('loaded weight from self supervised training')
    sys.stdout.flush()

    loss = nn.MSELoss()
    running_loss_val = 0

    model.eval()
    
    with torch.no_grad():
        for step, batch in  enumerate(loader_val):
            X_ray, stats, position_label = batch
            X_ray = X_ray.to(DEVICE).to(torch.float32)
            stats = stats.to(DEVICE).to(torch.float32)
            position_label = position_label.to(DEVICE).to(torch.float32)

            pred = model(stats, X_ray)
            
            val_loss_ = loss(pred, position_label)
            running_loss_val += val_loss_.item()
            print(f'validation Loss: {val_loss_}')
            sys.stdout.flush()

        print(f'Avg MSE loss: {running_loss_val/len(loader_val)} Denormalized: {(running_loss_val/len(loader_val))*568}')
        sys.stdout.flush()
        torch.cuda.empty_cache()

if __name__=="__main__":
    args = parse_args()
    main(args)