# C-Arm Guidance
Official implementation of the paper "C-Arm Guidance: a Self-Supervised Approach to Automated Positioning During Stroke Thrombectomy" (ISBI 2025)

![model](assets/carm_arch.svg)

## Prerequisites
### Training & Testing
- Pytorch
- torchvision
- pillow
- tqdm
  
### GUI
in progress 

## Training
For training you'll need to provide a dataset composed of X-ray images. Feel free to either use our provided GUI or your own data (make sure to adapt the code)

For the best experience, make sure your data repository tree is as follows:
```bash
regression
├── case_00000/
   └── *.png
├── case_00001/
   └── *.png
├── .
├── .
├── case_N/
   └── *.png
```
```bash
Landmarks
├── 20/
   └── *.png
├── 19/
   └── *.png
├── .
├── .
├── 1/
   └── *.png 
```
Also, the dataset class assumes to have an annotation .csv files with the following columns:
#### regression
annotations.csv
|   column      |      example          | Description                                |
|---------------|------------------------|------------------------------------------|
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
    
#### classifier
annotations.csv
|   column      |      example      | Description                                |
|---------------|-------------------|-----------------------------------------------|
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

### Pretext Task (Regression)
To train the first task (regression) navigate to the root directory and run the following script
```
bash ./scripts/train_self_supervised.sh
```
Note that you can customize the training experiment using the arguments provided in `./src/train_self_supervised.py` by simply passing them to `./scripts/train_self_supervised.sh`

This script will save checkpoints at every epoch in `./LOG_DIR/EXP_NAME/checkpoints/` (all of these variable should be provided in `train_self_supervised.sh`)

### Downstream Task (Classification)
To finetune your model on the classification task, navigate to the root directory and run the following script
```
bash ./scripts/train_classifier.sh
```
Note that you can customize the training experiment using the arguments provided in `./src/train_classifier.py` by simply passing them to `./scripts/train_classifier.sh` (make sure to locate the checkpoint from the pretraining phase)

This script will save checkpoints at every epoch in `./LOG_DIR/EXP_NAME/checkpoints/` (all of these variable should be provided in `train_classifier.sh`)

#### Training Experiments
You can customize the training experiment using the arguments provided in each script by simply passing them to `./scripts/train_classifier.sh` or `./scripts/train_self_supervised.sh`, for example:
```
--pretrained_weights="imagenet"
```
The following are the main experiments ran in the paper:
| Argument       | Experiment    |
|----------------|---------------|
| `--pretrained_weights="position"` | Determine the classifier pretraining |
| `--pretrained_weights="imagenet"` | Determine the classifier pretraining |
| `--pretrained_weights="none"` | Determine the classifier pretraining |
| `--linear_probing` | Implement linear probing and freeze all layers except the last linear ones |
| `--remove_patient_stats` | Ablation study to not include the patient demographics |
