import os
import torch
import pandas as pd
from PIL import Image
from utils import *

class Landmark_dataset(torch.utils.data.Dataset):
    """
    Landmark classification dataset

    This class expects the data to be stored in a directory with the following convention:
        
        Landmarks/
            |_ 20/
                |_ *.png
            |_ 19/
                |_ *.png
            .
            .
            .
            |_ 1/
                |_ *.png

    Along with an annotations .csv file that describes the data as follows:
    annotations.csv
     ________________________________________________________________________________
    |   column      |      example      | Description                                |
    ---------------------------------------------------------------------------------
    |  case_number  |     case-10065    | uniques case ID                            |
    |   filename    | root/20/10065.png | file path for the Xray .png image          |
    |      x        |      -232.5       | x-position of the image in the CT          |
    |      y        |       -34.5       | y-position of the image in the CT          |
    |      z        |       208.2       | z-position of the image in the CT          |
    |     part      |       upper       | Xray belongs to the 'upper' or 'lower' CT  |
    |   age_years   |        66         | patient age                                |
    |   sex_code    |       Male        | patient sex                                |
    |cadaver_weight |       73.0        | cadaver weight (kg)                        |
    |cadaver_length |      174.0        | cadaver legnth (m)                         |
    |     mode      |      train        | 'train' or 'test'                          |
    |   landmark    |       20          | landmark label [1-20]                      |
    -----------------------------------------------------------------------------------

    """
    
    def __init__(self,
                 root_annotations, # csv annotations path
                 augmentation=True,
                 mode='train',
                 size=[224,224]):
        
        super(Landmark_dataset, self).__init__()
        
        self.root_annotations = root_annotations
        self.augmentation = augmentation
        self.mode = mode
        self.size = size
        self.transform = transform(size=self.size, augmentation=self.augmentation)

        self.annotations = pd.read_csv(f'{root_annotations}/annotations.csv', index_col=0)
        
        # save train set for tracking
        train_set = self.annotations[self.annotations['mode'] == 'train'].copy()
        
        self.annotations = self.annotations[self.annotations['mode'] == self.mode]

        # preprocessing (use trainset statistics)
        self.annotations['sex_code'] = self.annotations['sex_code'].map({'Male':0, 'Female':1})
        self.annotations['age_years'] = (self.annotations['age_years']-train_set['age_years'].mean())/train_set['age_years'].std()
        self.annotations['cadaver_weight'] = (self.annotations['cadaver_weight']-train_set['cadaver_weight'].mean())/train_set['cadaver_weight'].std()
        self.annotations['cadaver_length'] = (self.annotations['cadaver_length']-train_set['cadaver_length'].mean())/train_set['cadaver_length'].std()
        self.annotations['x'] = self.annotations['x']/train_set['x'].max()
        self.annotations['y'] = self.annotations['y']/train_set['y'].max()
        self.annotations['z'] = self.annotations['z']/train_set['z'].max()

    def __getitem__(self, index):
        sample = self.annotations.iloc[index]
        
        X_ray = Image.open(os.path.join(sample.filename), 'r').convert('L')
        X_ray = self.transform(X_ray).repeat(3,1,1)

        age = torch.tensor([sample['age_years']])
        weight = torch.tensor([sample['cadaver_weight']])
        height = torch.tensor([sample['cadaver_length']])
        sex = torch.nn.functional.one_hot(torch.tensor([sample['sex_code']]), 2).view(-1)
        
        label = torch.tensor(sample['landmark'])

        return (X_ray, torch.cat([age,weight,height,sex]), label)

    def __len__(self):
        return len(self.annotations)
    

class Positional_dataset(torch.utils.data.Dataset):
    """
    Self supervised regression dataset

    This class expects the data to be stored in a directory with the following convention:
    Each patient CT is densely samples and saved in a seperate dir
        
        regression/
            |_ case_0/
                |_ *.png
            |_ case_1/
                |_ *.png
            .
            .
            .
            |_ case_n/
                |_ *.png

    Along with an annotations .csv file that describes the data as follows:
    annotations.csv
     ________________________________________________________________________________
    |   column      |      example          | Description                                |
    ---------------------------------------------------------------------------------
    |  case_number  |     case-10065        | uniques case ID                            |
    |   filename    | root/case_10065/5.png | file path for the Xray .png image          |
    |      x        |      -232.5           | x-position of the image in the CT          |
    |      y        |       -34.5           | y-position of the image in the CT          |
    |      z        |       208.2           | z-position of the image in the CT          |
    |     part      |       upper           | Xray belongs to the 'upper' or 'lower' CT  |
    |   age_years   |        66             | patient age                                |
    |   sex_code    |       Male            | patient sex                                |
    |cadaver_weight |       73.0            | cadaver weight (kg)                        |
    |cadaver_length |      174.0            | cadaver legnth (m)                         |
    |     mode      |      train            | 'train' or 'test'                          |
    -------------------------------------------------------------------------------------
    
    """
    def __init__(self,
                 root_annotations, # annotations root
                 augmentation=True,
                 mode='train',
                 size=[256,256]):
        
        super(Positional_dataset, self).__init__()
        
        self.root_annotations = root_annotations
        self.augmentation = augmentation
        self.mode = mode
        self.size = size
        self.transform = transform(size=self.size, augmentation=self.augmentation)

        self.annotations = pd.read_csv(f'{root_annotations}/full_dataset.csv', index_col=0)
        
        # save train set for tracking
        train_set = self.annotations[self.annotations['mode'] == 'train'].copy()
        
        self.annotations = self.annotations[self.annotations['mode'] == self.mode]

        # preprocessing (use trainset statistics)
        self.annotations['sex_code'] = self.annotations['sex_code'].map({'Male':0, 'Female':1})
        self.annotations['age_years'] = (self.annotations['age_years']-train_set['age_years'].mean())/train_set['age_years'].std()
        self.annotations['cadaver_weight'] = (self.annotations['cadaver_weight']-train_set['cadaver_weight'].mean())/train_set['cadaver_weight'].std()
        self.annotations['cadaver_length'] = (self.annotations['cadaver_length']-train_set['cadaver_length'].mean())/train_set['cadaver_length'].std()
        self.annotations['x'] = self.annotations['x']/train_set['x'].max()
        self.annotations['y'] = self.annotations['y']/train_set['y'].max()
        self.annotations['z'] = self.annotations['z']/train_set['z'].max()
        
    def __getitem__(self, index):
        sample = self.annotations.iloc[index]
        
        X_ray = Image.open(os.path.join(sample.filename), 'r').convert('L')
        X_ray = self.transform(X_ray).repeat(3,1,1)

        age = torch.tensor([sample['age_years']])
        weight = torch.tensor([sample['cadaver_weight']])
        height = torch.tensor([sample['cadaver_length']])
        sex = torch.nn.functional.one_hot(torch.tensor([sample['sex_code']]), 2).view(-1)
        x = torch.tensor([sample['x']])
        y = torch.tensor([sample['y']])
        z = torch.tensor([sample['z']])

        return (X_ray, torch.cat([age,weight,height,sex]), torch.cat([x, y, z]))

    def __len__(self):
        return len(self.annotations)