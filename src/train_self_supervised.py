import torch.nn as nn
import torch
from dataset import Positional_dataset
from models import *
import argparse
import os
import sys
from tqdm import tqdm

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training self-supervised model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="root/logs/self_supervised",
        help=("Be careful when using deep green or black diamond")
    )
    parser.add_argument(
        "--annotations_dir",
        type=str,
        default=None,
        required=True,
        help=("Annotations folder")
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        required=True,
        help=("the name of the directory in log_dir")
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='resnet34',
        required=False,
        help=("the name of the backbone used in training")
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def main(args):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpu_count = torch.cuda.device_count()

    # logging stuff
    log_dir = f'{args.log_dir}/{args.exp_name}'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f'{log_dir}/checkpoints', exist_ok=True)
    
    img_size = [256,256]

    dataset = Positional_dataset(augmentation=True, mode='train', size=img_size, root_annotations=args.annotations_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    dataset_val = Positional_dataset(augmentation=False, mode='test', size=img_size, root_annotations=args.annotations_dir)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    
    model = nn.DataParallel(positional_understanding_model(backbone=args.model_name))
    model = model.to(DEVICE).to(torch.float32)
        
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    model.train()

    print(f'Training {args.exp_name} for {args.epochs} epochs on {DEVICE} will start, using {gpu_count} GPUS!')
    loss_list = []
    loss_list_val = []

    for e in range(args.epochs):
        running_loss = 0
        running_loss_val = 0
        model.train()

        for i, batch in enumerate(tqdm(loader)):
            X_ray, stats, position_label = batch
            X_ray = X_ray.to(DEVICE).to(torch.float32)
            stats = stats.to(DEVICE).to(torch.float32)
            position_label = position_label.to(DEVICE).to(torch.float32)

            pred = model(stats, X_ray)
            
            loss_ = loss(pred, position_label)
            loss_.backward()

            running_loss += loss_.item()

            optimizer.step()
            optimizer.zero_grad()

            print(f'loss: {loss_}')
            sys.stdout.flush()

        loss_list.append(running_loss/len(loader))
        
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

        loss_list_val.append(running_loss_val/len(loader_val))

        print(f'finished epoch {e}\n----------------------------------------------------')
        sys.stdout.flush()

        torch.save({
            'model_state_dict': model.state_dict(),
            'train_loss': loss_list,
            'val_loss': loss_list_val,
        }, f'{log_dir}/checkpoints/chkpt_e{e}.pt')

        torch.cuda.empty_cache()

if __name__=="__main__":
    args = parse_args()
    main(args)