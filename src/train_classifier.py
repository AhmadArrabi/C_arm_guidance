import torch.nn as nn
import torch
from dataset import Landmark_dataset
from models import *
import argparse
import os
import sys
from tqdm import tqdm

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training classifier model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="root/logs/classifier",
        help=("make sure you're logging in the right dir")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        required=True,
        help=("Xray data folder")
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
        required=True,
        help=("the name of the model used in training")
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default="position",
        help=("choose between [position, imagenet, none]")
    )
    parser.add_argument(
        "--checkpoint_self_supervised",
        type=str,
        default="root/logs/self_supervised/experiment_1/checkpoints/chkpt_e100.pt",
        help=(".pth checkpoint file for the self supervised regression pretraining")
    )
    parser.add_argument(
        "--classifier_layers",
        type=int,
        default=1,
        help=("number of classifier layers that replace the regression head. choose between [1,2]")
    )
    parser.add_argument(
        "--linear_probing",
        action='store_true'
    )
    parser.add_argument(
        "--remove_patient_stats",
        action='store_true'
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

    dataset = Landmark_dataset(augmentation=True, mode='train', size=img_size, root_data=args.data_dir, root_annotations=args.annotations_dir)
    dataset_val = Landmark_dataset(augmentation=False, mode='test', size=img_size, root_data=args.data_dir, root_annotations=args.annotations_dir)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False)
    
    map_pretrained_weights = {'position': None,
                              'imagenet': 'DEFAULT',
                              'none': None}
    
    model = nn.DataParallel(positional_understanding_model(backbone=args.model_name, weights=map_pretrained_weights[args.pretrained_weights]), remove_patient_stats=args.remove_patient_stats)
    model = model.to(DEVICE).to(torch.float32)
    
    # load weights
    if args.pretrained_weights == 'position':
        checkpoint = torch.load(f'{args.checkpoint_self_supervised}')
        state_dict = checkpoint['model_state_dict']
        
        if args.remove_patient_stats:
            state_dict = {k: v for k, v in state_dict.items() if 'patient_processing' not in k}

        model.load_state_dict(state_dict)
        print('loaded weight from self supervised training')
        sys.stdout.flush()
        
    # add classifier head
    if args.classifier_layers == 1:
        # change last layers
        model.regression_head[-1] = nn.Linear(64, 20)
    elif args.classifier_layers == 2:
        # change last two layers (replace all of regression head)
        model.regression_head = nn.Sequential(nn.Linear(128, 64), 
                                              nn.ReLU(), 
                                              nn.Linear(64, 20))
    else:
        raise RuntimeError(f'can not retrain {args.classifier_layers} layers of classifier, choose 1 or 2')
    
    model.to(DEVICE)
    loss = nn.CrossEntropyLoss()

    if args.linear_probing:
        print('linear probing implemented')
        for param in model.parameters():
            param.requires_grad = False

        # unfreeze last layer (classification layer)        
        for param in model.regression_head.parameters():
            param.requires_grad = True

    if args.linear_probing:
        optimizer = torch.optim.Adam(model.regression_head.parameters())
    else:
        optimizer = torch.optim.Adam(model.parameters())

    print(f'Training {args.exp_name} for {args.epochs} epochs on {DEVICE} will start, using {gpu_count} GPUS!')
    loss_list = []
    loss_list_val = []

    for e in range(args.epochs):
        running_loss = 0
        running_loss_val = 0
        model.train()

        for i, batch in enumerate(tqdm(loader)):
            X_ray, stats, landmark_label = batch
            X_ray = X_ray.to(DEVICE).to(torch.float32)
            stats = stats.to(DEVICE).to(torch.float32)
            landmark_label = landmark_label.to(DEVICE)

            pred = model(stats, X_ray)

            loss_ = loss(pred, landmark_label)
            loss_.backward()

            running_loss += loss_.item()

            optimizer.step()
            optimizer.zero_grad()

            print(f'Loss: {loss_}')
            sys.stdout.flush()

        loss_list.append(running_loss/len(loader))

        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(loader_val):
                X_ray, stats, landmark_label = batch
                X_ray = X_ray.to(DEVICE).to(torch.float32)
                stats = stats.to(DEVICE).to(torch.float32)
                landmark_label = landmark_label.to(DEVICE)

                pred = model(stats, X_ray)

                val_loss_ = loss(pred, landmark_label)
                running_loss_val += val_loss_.item()
                
                print(f'Validation Loss: {val_loss_}')
                sys.stdout.flush()

        loss_list_val.append(running_loss_val/len(loader_val))
        print(f'finished epoch {e}\n----------------------------------------------------')
        sys.stdout.flush()

        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': loss_list,
            'val_loss': loss_list_val,
        }, f'{log_dir}/checkpoints/chkpt_e{e}.pt')

        torch.cuda.empty_cache()

if __name__=="__main__":
    args = parse_args()
    main(args)