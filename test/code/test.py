import numpy as np
from pathlib import Path
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms 

from emonet.models import EmoNet
from emonet.data import AffectNet
from emonet.data_augmentation import DataAugmentor
from emonet.metrics import CCC, PCC, RMSE, SAGR, ACC
from emonet.evaluation import evaluate, evaluate_flip

torch.backends.cudnn.benchmark =  True

#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--nclasses', type=int, default=8, choices=[5,8], help='Number of emotional classes to test the model on. Please use 5 or 8.')
args = parser.parse_args()

# Parameters of the experiments
n_expression = args.nclasses
batch_size = 32
n_workers = 16
device = 'cuda:0'
image_size = 256
subset = 'test'
metrics_valence_arousal = {'CCC':CCC, 'PCC':PCC, 'RMSE':RMSE, 'SAGR':SAGR}
metrics_expression = {'ACC':ACC}

# Create the data loaders
transform_image = transforms.Compose([transforms.ToTensor()])
transform_image_shape_no_flip = DataAugmentor(image_size, image_size)

flipping_indices = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 26, 25, 24, 23, 22,21, 20, 19, 18, 17, 27, 28, 29, 30, 35, 34, 33, 32, 31, 45,44, 43, 42, 47, 46, 39, 38, 37, 36, 41, 40, 54, 53, 52, 51,50, 49, 48, 59, 58,57, 56, 55, 64, 63,62, 61, 60, 67, 66,65]
transform_image_shape_flip = DataAugmentor(image_size, image_size, mirror=True, shape_mirror_indx=flipping_indices, flipping_probability=1.0)


# Loading the model 
state_dict_path = Path(__file__).parent.joinpath('pretrained', f'emonet_{n_expression}.pth')

print(f'Loading the model from {state_dict_path}.')
state_dict = torch.load(str(state_dict_path), map_location='cpu')
state_dict = {k.replace('module.',''):v for k,v in state_dict.items()}
net = EmoNet(n_expression=n_expression).to(device)
net.load_state_dict(state_dict, strict=False)
net.eval()

# =======changin' ============
from skimage import io
import cv2
from torchvision import transforms
import PIL
import os

def tensor_to_image(tensor,out_dir='./testing'):
    if len(tensor.shape)==4:
        raise NotImplemented
    else:
        # choose filename
        idx = len([f for f in os.listdir(out_dir) if f.starts_with('test')])
        fname = 'test_%d.png'%idx
        pth = os.path.join(out_dir,fname)

        image = torch.permute(tensor,(1,2,0))
        image = (255*image).cpu().numpy()
        
        cv2.imwrite(pth,image)

with torch.no_grad():
    image_file = '/home/spock-the-wizard/cmu/sg/visa_photo.jpg'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(256),
    ])

    image = cv2.imread(image_file)
    image = transform(image)

    tensor_to_image(image)

    image = torch.unsqueeze(image,dim=0)
    
    out = net(image)
  
    print('Saved image')

    
    heatmap = out['heatmap']
    exp = out['expression']
    val = out['valence']
    aro = out['arousal']
    # prin
    # import pdb;pdb.set_trace()
    print(out)

    # import pdb;pdb.set_trace()

