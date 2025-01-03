import torchvision.models as models
import torch.nn as nn

class ResNet34(nn.Module):
    def __init__(self, out_classes=20):
        super().__init__()

        self.model = models.resnet34(weights='DEFAULT')
        self.model.fc = nn.Linear(in_features=512, out_features=out_classes)

    def forward(self, x):
        return self.model(x)
    
class ConvNext_base(nn.Module):
    def __init__(self, out_classes=20):
        super().__init__()

        self.model = models.convnext_base(weights='DEFAULT')
        self.model.classifier[-1] = nn.Linear(in_features=1024, out_features=out_classes)

    def forward(self, x):
        return self.model(x)
    
class VIT_b_16(nn.Module):
    def __init__(self, out_classes=20):
        super().__init__()

        self.model = models.vit_b_16(weights='DEFAULT')
        self.model.heads.head = nn.Linear(in_features=768, out_features=out_classes)

    def forward(self, x):
        return self.model(x)
    
class ConvBN_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding) :
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.block(x)
    
class Linear_block(nn.Module):
    def __init__(self, in_features, out_features) :
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Dropout()
        )

    def forward(self, x):
        return self.block(x)

class PoseNet(nn.Module):
    def __init__(self,
                 in_channels = 1,
                 n_output = 6):
        super().__init__()

        self.CNN_network = nn.Sequential(
            ConvBN_block(in_channels, 16, 3, 2, 1),
            ConvBN_block(16, 16, 3, 2, 1),

            ConvBN_block(16, 32, 3, 2, 1),
            ConvBN_block(32, 32, 3, 2, 1),
            
            ConvBN_block(32, 64, 3, 1, 1),
            ConvBN_block(64, 64, 3, 2, 1),
            
            ConvBN_block(64, 128, 3, 1, 1),
            ConvBN_block(128, 128, 3, 2, 1),
            
            ConvBN_block(128, 128, 4, 1, 0),
            
            nn.Flatten(),
            Linear_block(128, 2048),
            Linear_block(2048, 1024),

            nn.Linear(1024, n_output)
        )

    def forward(self, x):
        return self.CNN_network(x)
    
class positional_understanding_model(nn.Module):
    def __init__(self,
                 backbone='resnet34',
                 weights='DEFAULT',
                 emb_dim=128,
                 kdim=128,
                 vdim=128,
                 nheads=8,
                 outs=3,
                 remove_patient_stats=False):
        
        super().__init__()

        self.remove_patient_stats = remove_patient_stats

        if backbone == 'resnet34':
            resnet = models.resnet34(weights=weights) #512
            self.backbone_fc = nn.Sequential(nn.Linear(512, kdim), nn.ReLU())
        elif backbone == 'resnet101':
            resnet = models.resnet101(weights=weights) #2048
            self.backbone_fc = nn.Sequential(nn.Linear(2048, kdim), nn.ReLU())

        print(f'loaded {backbone} with {weights} weights')
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) #feature extractor (512-2048, h/32, w/32)
        
        if not self.remove_patient_stats:
            self.patient_processing_module = patient_processing_module(emb_dim=emb_dim,
                                                                    nheads=nheads,
                                                                    kdim=kdim,
                                                                    vdim=vdim)
        
        self.regression_head = nn.Sequential(
                                        nn.Linear(128, 64),
                                        nn.ReLU(),
                                        nn.Linear(64, outs)
                                        )


    def forward(self, patient_stats, image):
        batch_size = image.shape[0]
        z_image = self.backbone(image)
        z_image = z_image.view(batch_size, z_image.shape[1])
        z_image = self.backbone_fc(z_image)
        
        if not self.remove_patient_stats:
            r = self.patient_processing_module(patient_stats, z_image)
            p = z_image*r
            p = self.regression_head(p)
            return p
        
        else:
            return self.regression_head(z_image)
    
class patient_processing_module(nn.Module):
    def __init__(self,
                 emb_dim=128,
                 nheads=8,
                 kdim=128,
                 vdim=128):
        
        super().__init__()

        self.linear_embedding = nn.Sequential(
            nn.Linear(5, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(64, emb_dim),
            nn.BatchNorm1d(emb_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2)
            )
        
        self.cross_attention_block = nn.MultiheadAttention(emb_dim, nheads, kdim=kdim, vdim=vdim)

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.ReLU())
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.dropout2 = nn.Dropout(p=0.2)

    def forward(self, patient_stats, z_image):
        patient_stats = self.linear_embedding(patient_stats)
        r, _ = self.cross_attention_block(patient_stats, z_image, z_image)
        r = self.dropout1(r)
        r = self.layer_norm1(patient_stats + r)
        r = self.layer_norm2(self.dropout2(self.fc1(r))) + r
        return r


