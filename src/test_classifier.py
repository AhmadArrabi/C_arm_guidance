import torch
import numpy as np
from models import *
from dataset import Landmark_dataset
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import sys
import argparse

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Training classifier model")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
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
        default="root/logs/classifier/experiment_1/checkpoints/chkpt_e100.pt",
        help=(".pth checkpoint file for the self supervised regression pretraining")
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default='resnet34',
        required=False,
        help=("the name of the backbone used in training")
    )
    parser.add_argument(
        "--classifier_layers",
        type=int,
        default=1,
        help=("number of classifier layers that replace the regression head. choose between [1,2]")
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
    
    dataset_val = Landmark_dataset(augmentation=False, mode='test', size=[256,256], root_annotations=args.annotations_dir)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=True)
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        model = nn.DataParallel(positional_understanding_model(backbone=args.model_name, remove_patient_stats=args.remove_patient_stats))
        
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
        
        model = model.to(DEVICE).to(torch.float32)

        checkpoint = torch.load(f'{args.checkpoint_path}', map_location=DEVICE)
        state_dict = checkpoint['model_state_dict']

        model.load_state_dict(state_dict)
        model.eval()

        labels = []
        preds = []

        for step, batch in enumerate(loader_val):
            X_ray, stats, landmark_label = batch
            X_ray = X_ray.to(DEVICE).to(torch.float32)
            stats = stats.to(DEVICE).to(torch.float32)
            landmark_label = landmark_label.to(DEVICE)
            
            pred = model(stats, X_ray)
            probabilities = F.softmax(pred, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)  # predicted_classes shape: [B]

            labels.extend(landmark_label.cpu().numpy())
            preds.extend(predicted_classes.cpu().numpy())
        
        labels = np.array(labels)
        preds = np.array(preds)

        conf_matrix = confusion_matrix(labels, preds)
        report = classification_report(labels, preds, target_names=[f'{i}' for i in range(20)])
        
        print("Classification Report:")
        sys.stdout.flush()
        print(report)
        sys.stdout.flush()

if __name__=="__main__":
    main()
