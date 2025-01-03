from torchvision import transforms
import pandas as pd

def transform(size, augmentation=True):
    if augmentation:
        return transforms.Compose([
            transforms.Resize(size=tuple(size)),
            transforms.ColorJitter(0.15, 0.15, 0.15),
            transforms.RandomPosterize(p=0.15, bits=4),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(size=tuple(size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])